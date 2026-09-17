# -*- coding: utf-8 -*-
"""smy_cd 目标编码——模型级增量快速验证（单一 LightGBM，非 DART）

不改动 assemble.py / 01~10 正式链路，只是调用现成的 assemble.build_train()
拿到基线 401 维矩阵，再拼上本脚本内重算的 smy_te 列做对比：
  baseline_401       现有 401 维特征
  plus_smy_te_402    401 维 + smy_te

评估口径与线上一致：5 折 OOF 后取 top1_f1（见 config.top1_f1）。
用单模型代替全量三族融合，目的是用较低成本（预计 10~20 分钟量级，
LightGBM 非 DART + early stopping）判断值不值得投入 2.5 小时全量重训。

用法: PYTHONPATH=code python code/explore/smy_te_lift_check.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P, FOLD_SEED, N_FOLDS, LGB_BASE, top1_f1
import assemble

t0 = time.time()

tr, X, y, cols = assemble.build_train()
print(f"基线矩阵: {X.shape}")

# ---------------- 重算 smy_cd fold-safe 加权目标编码 ----------------
txn = pl.read_parquet(P("txn.parquet"), columns=["card_no", "smy_cd"])
smy_all = (txn.group_by(["card_no", "smy_cd"]).len()
           .rename({"len": "n"}).to_pandas())
smy_all["card_no"] = smy_all["card_no"].astype(str)
del txn

cards_df = tr[["card_no"]].copy()
cards_df["card_no"] = cards_df["card_no"].astype(str)
card_pos = {c: i for i, c in enumerate(cards_df["card_no"].values)}
smy_idx = smy_all.merge(cards_df.assign(label=y), on="card_no", how="inner")
global_mean = y.mean()
smy_te = np.full(len(cards_df), global_mean, dtype=np.float64)

folds = list(StratifiedKFold(N_FOLDS, shuffle=True,
                             random_state=FOLD_SEED).split(X, y))
for tr_idx, va_idx in folds:
    tr_cards = set(cards_df.iloc[tr_idx]["card_no"])
    va_cards = set(cards_df.iloc[va_idx]["card_no"])
    ref = smy_idx[smy_idx["card_no"].isin(tr_cards)].copy()
    ref["wl"] = ref["label"] * ref["n"]
    code_stat = ref.groupby("smy_cd").agg(w=("wl", "sum"), n=("n", "sum"))
    code_rate = (code_stat["w"] / code_stat["n"]).to_dict()
    va = smy_idx[smy_idx["card_no"].isin(va_cards)].copy()
    va["rate"] = va["smy_cd"].map(code_rate).fillna(global_mean)
    va["wn"] = va["rate"] * va["n"]
    agg = va.groupby("card_no").agg(wn_sum=("wn", "sum"), n_sum=("n", "sum"))
    agg["te"] = agg["wn_sum"] / agg["n_sum"]
    for c, v in agg["te"].items():
        smy_te[card_pos[c]] = v

X_plus = X.copy()
X_plus["smy_te"] = smy_te.astype(np.float32)
print(f"实验矩阵: {X_plus.shape}")


def run_oof(Xmat, tag):
    oof = np.zeros(len(y), dtype=np.float64)
    for k, (tr_idx, va_idx) in enumerate(folds):
        dtr = lgb.Dataset(Xmat.iloc[tr_idx], y[tr_idx])
        dva = lgb.Dataset(Xmat.iloc[va_idx], y[va_idx])
        m = lgb.train(LGB_BASE, dtr, num_boost_round=3000,
                      valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va_idx] = m.predict(Xmat.iloc[va_idx], num_iteration=m.best_iteration)
        print(f"  [{tag}] fold{k} best_iter={m.best_iteration} {time.time()-t0:.0f}s")
    f1, tp = top1_f1(y, oof)
    print(f"[{tag}] OOF top1_f1={f1:.4f} (tp={tp})")
    return oof, f1, tp


oof_base, f1_base, tp_base = run_oof(X, "baseline_401")
oof_plus, f1_plus, tp_plus = run_oof(X_plus, "plus_smy_te_402")

print("\n" + "=" * 60)
print(f"baseline(401维): top1_f1={f1_base:.4f}  tp={tp_base}")
print(f"+smy_te(402维):  top1_f1={f1_plus:.4f}  tp={tp_plus}")
print(f"命中数变化: {tp_plus - tp_base:+d} 张")
print(f"\n全部完成 {time.time()-t0:.0f}s")
