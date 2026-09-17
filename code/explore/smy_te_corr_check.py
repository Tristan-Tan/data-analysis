# -*- coding: utf-8 -*-
"""smy_cd 目标编码 与已有 299 维特征的最高相关性验证（仿 silence 特征验证方法论）

field_value_check.py 已测得 smy_cd 加权目标编码单变量 AUC = 0.7714，
但现有 05_features_v4.py 已对 Top95 摘要码做了 one-hot share 展开。
本脚本判断：目标编码这一列是模型完全无法表达的新信息（类比 silence 与已有
特征最高相关仅 0.491），还是现有 smy_*/smy2_* one-hot 已经隐含覆盖的冗余信息。

前置: 已跑过 01_parse_raw.py / 05_features_v4.py
用法: PYTHONPATH=code python code/explore/smy_te_corr_check.py
耗时: 约 1~2 分钟（smy_cd 目标编码本身很快，5 折合计数十秒）
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P, FOLD_SEED, N_FOLDS

t0 = time.time()

train = pl.read_parquet(P("train.parquet")).select(["card_no", "label"]).to_pandas()
train["card_no"] = train["card_no"].astype(str)
train["label"] = train["label"].astype(int)
y = train["label"].values

# ---------------- 重算 smy_cd fold-safe 加权目标编码 ----------------
txn = pl.read_parquet(P("txn.parquet"), columns=["card_no", "smy_cd"])
smy_card = (txn.group_by(["card_no", "smy_cd"]).len()
            .rename({"len": "n"}).to_pandas())
smy_card["card_no"] = smy_card["card_no"].astype(str)
del txn

cards_df = train[["card_no", "label"]].reset_index(drop=True)
card_pos = {c: i for i, c in enumerate(cards_df["card_no"].values)}
smy_idx = smy_card.merge(cards_df, on="card_no", how="inner")
global_mean = y.mean()
oof = np.full(len(cards_df), global_mean, dtype=np.float64)

skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED)
row_idx = np.arange(len(cards_df))
for tr_idx, va_idx in skf.split(row_idx, y):
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
        oof[card_pos[c]] = v

print(f"smy_cd 目标编码重算完成 {time.time()-t0:.0f}s")

# ---------------- 与已有 299 维特征做相关性对比 ----------------
feat = pd.read_parquet(P("features_v4.parquet"))
feat["card_no"] = feat["card_no"].astype(str)
feat = feat.set_index("card_no").reindex(cards_df["card_no"]).reset_index()

num_cols = [c for c in feat.columns if c != "card_no"
            and pd.api.types.is_numeric_dtype(feat[c])]

corrs = {}
for c in num_cols:
    v = feat[c].fillna(0).values.astype(np.float64)
    if v.std() == 0:
        continue
    corrs[c] = abs(np.corrcoef(v, oof)[0, 1])
corrs = pd.Series(corrs).sort_values(ascending=False)

print(f"\n与全部 {len(num_cols)} 维已有特征相关性 Top10（判断是否为全新信息）：")
print(corrs.head(10))

smy_related = [c for c in num_cols if c.startswith("smy_") or c.startswith("smy2_")]
corrs_smy = corrs[corrs.index.isin(smy_related)].sort_values(ascending=False)
print(f"\n仅与现有 {len(smy_related)} 个 smy one-hot share 特征的相关性 Top10"
      "（判断 one-hot 是否已隐含覆盖）：")
print(corrs_smy.head(10))

print(f"\n全部完成 {time.time()-t0:.0f}s")
