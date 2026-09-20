# -*- coding: utf-8 -*-
"""实验 3：exp5 的表达方式 —— 绝对关联计数 vs 平滑比率

【与已失败的 silence 屏蔽的区别】
silence 屏蔽是**删掉整个信息族**；本实验是**换一种表达**：同一份对手关联
信息，去掉随参考集规模线性放大的绝对计数（nf_max/ncp_f1/ncp_f2/amt_f），
只保留与规模无关的平滑比率（rate_max/rate_wavg/txnf_share）与行为特征。

依据：PSI 诊断中 *_nf_max 系列 PSI 高达 1.6~1.7，是全部 405 维里漂移最严重
的一组；而 A 榜的 fold-matched 修正本身就是为了对齐这类计数的参考集规模。

【段一 诊断：巨大公共对手是否支配统计】
对每个交易对手计算「超额欺诈率」=(该对手关联的欺诈卡数/该对手度数)/全局
欺诈率。若高度数对手的该值 ≈ 1，说明它们只是"大"而非"可疑"，nf_max 在其
上取到的极值即为噪声。
⚠ 该诊断使用全量 train label，**仅用于描述**，不参与特征构造，也不用于
筛选列——筛选规则（去掉绝对计数列）是先验的、不依赖 label 的，故无泄漏。

【段二 对照实验】唯一变量为 exp5 的表达方式，模型/折/超参/seed 全部相同：
  baseline : 现有 402 维
  exp3     : 402 维 - 32 列绝对计数 = 370 维

前置: 10_train_models.py testb 已跑过
用法: PYTHONPATH=code python code/labB/run_exp3.py testb
耗时: 约 15~20 分钟
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (P, FOLD_SEED, N_FOLDS, LGB_BASE, LGBN_EXTRA, TOP_FRAC,
                    top1_f1)
from lab_common import (LP, LAB_PRED, R, swap_report, boundary_auc)
import assemble

t0 = time.time()
TARGET = sys.argv[1] if len(sys.argv) > 1 else "testb"
TAG = "exp3"
FAMILIES = ["dart", "abthin_lgbn", "cat"]
# exp5 的 7 个统计量中，随参考集规模线性放大的绝对量
ABS_SUFFIX = ("_nf_max", "_ncp_f1", "_ncp_f2", "_amt_f")
RATIO_SUFFIX = ("_rate_max", "_rate_wavg", "_txnf_share")

# ==================== 段一：公共对手支配性诊断 ====================
print("=" * 74)
print("【段一】巨大公共对手是否支配 exp5 统计（描述性诊断，不用于筛选）")
tr_lab = pl.read_parquet(P("train.parquet")).select(["card_no", "label"])
base_rate = tr_lab["label"].mean()
print(f"  全局欺诈率 {base_rate:.4%}")

for name in ("edge_acc2", "edge_nm2"):
    p = P(f"{name}.parquet")
    if not os.path.exists(p):
        print(f"  缺少 {name}.parquet，跳过")
        continue
    e = pl.read_parquet(p, columns=["card_no", "key"]).unique()
    g = (e.join(tr_lab, on="card_no", how="inner")
         .group_by("key")
         .agg([pl.len().alias("deg"),
               pl.col("label").sum().alias("nf")]))
    g = g.with_columns(
        (pl.col("nf") / pl.col("deg") / base_rate).alias("excess"))
    d = g.to_pandas()
    print(f"\n  -- {name}：train 内有关联的对手 {len(d):,} 个 --")
    bins = [(2, 10), (10, 100), (100, 1000), (1000, 10000), (10000, 10 ** 9)]
    print(f"    {'度数区间':<16}{'对手数':>10}{'覆盖卡次':>12}"
          f"{'平均nf':>10}{'超额欺诈率':>12}")
    for lo, hi in bins:
        s = d[(d["deg"] >= lo) & (d["deg"] < hi)]
        if not len(s):
            continue
        print(f"    [{lo},{hi}){'':<6}{len(s):>10,}{int(s['deg'].sum()):>12,}"
              f"{s['nf'].mean():>10.1f}{s['excess'].mean():>12.2f}")
    top = d.nlargest(5, "nf")[["deg", "nf", "excess"]]
    print("    nf 最大的 5 个对手（nf_max 的实际来源）：")
    for _, r in top.iterrows():
        print(f"      度数={int(r['deg']):>7,}  关联欺诈卡={int(r['nf']):>5,}"
              f"  超额欺诈率={r['excess']:.2f}")
print(f"\n  判读：若 nf 最大的那几个对手度数极大而超额欺诈率≈1，"
      f"说明 nf_max 主要由公共对手的规模驱动，而非风险信号")

# ==================== 段二：对照实验 ====================
print("\n" + "=" * 74)
print("【段二】去掉绝对计数后的对照实验")
tr, Xtr, y, cols = assemble.build_train()
te, Xte_list = assemble.build_test_folds(N_FOLDS, cols)

abs_cols = [c for c in cols if c.endswith(ABS_SUFFIX)]
ratio_cols = [c for c in cols if c.endswith(RATIO_SUFFIX)]
keep = [c for c in cols if c not in set(abs_cols)]
print(f"  exp5 绝对计数列 {len(abs_cols)} 个（将剔除）："
      f"{abs_cols[:4]} ...")
print(f"  exp5 平滑比率列 {len(ratio_cols)} 个（保留）")
print(f"  基线 {len(cols)} 维 -> exp3 {len(keep)} 维   {time.time()-t0:.0f}s")
assert len(abs_cols) == 32, f"绝对计数列应为 32 个，实得 {len(abs_cols)}"

folds = list(StratifiedKFold(N_FOLDS, shuffle=True,
                             random_state=FOLD_SEED).split(Xtr, y))
params = dict(LGB_BASE, seed=FOLD_SEED)
params.update(LGBN_EXTRA)


def train(use_cols, tag):
    oof = np.zeros(len(Xtr))
    pte = np.zeros(len(Xte_list[0]))
    hits = []
    for k, (ti, vi) in enumerate(folds):
        m = lgb.train(params, lgb.Dataset(Xtr.iloc[ti][use_cols], y[ti]), 3000,
                      valid_sets=[lgb.Dataset(Xtr.iloc[vi][use_cols], y[vi])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[vi] = m.predict(Xtr.iloc[vi][use_cols],
                            num_iteration=m.best_iteration)
        pte += m.predict(Xte_list[k][use_cols],
                         num_iteration=m.best_iteration) / N_FOLDS
        kk = int(len(vi) * TOP_FRAC)
        hits.append(int(y[vi][np.argsort(-oof[vi])[:kk]].sum()))
        print(f"  [{tag}] fold{k} best_iter={m.best_iteration} "
              f"折内 top1% 命中 {hits[-1]}/{kk}   {time.time()-t0:.0f}s")
    return oof, pte, hits


oof_b, te_b, fh_b = train(cols, "baseline")
oof_e, te_e, fh_e = train(keep, TAG)

_, tp_b = top1_f1(y, oof_b)
_, tp_e = top1_f1(y, oof_e)
print("\n" + "=" * 74)
print(f"【1】全局 OOF Top1%：baseline {tp_b}/3000 -> {TAG} {tp_e}/3000"
      f"（{tp_e - tp_b:+d}）")
print(f"     各折：baseline {fh_b}  合计 {sum(fh_b)}")
print(f"           {TAG}      {fh_e}  合计 {sum(fh_e)}")
print(f"     逐折差: {[e - b for e, b in zip(fh_e, fh_b)]}")

print("\n【2/3】OOF 侧换卡与净收益")
swap_report(oof_b, oof_e, y=y, label="OOF")

print("\n【5】边界区分能力")
boundary_auc(y, oof_b, oof_e, 1500, 6000)
boundary_auc(y, oof_b, oof_e, 2500, 6000)

print(f"\n【4】{TARGET} 侧换卡（单模型对照）")
swap_report(te_b, te_e, y=None, label=f"{TARGET} 单模型")

np.save(LP(f"oof_{TAG}_{TARGET}.npy"), R(oof_e))
np.save(LP(f"te_{TAG}_{TARGET}.npy"), R(te_e))

print("\n【融合候选】三族 + exp3 作为第四族（族内平均后固定族权重等权）")
if not all(os.path.exists(P(f"te_{m}.npy")) for m in FAMILIES):
    print("  缺少正式 te_*.npy，跳过")
else:
    fte = {m: np.load(P(f"te_{m}.npy")) for m in FAMILIES}
    foof = {m: np.load(P(f"oof_{m}.npy")) for m in FAMILIES}
    b_oof = R(np.mean([R(foof[m]) for m in FAMILIES], axis=0))
    b_te = R(np.mean([R(fte[m]) for m in FAMILIES], axis=0))
    n_oof = R(np.mean([R(foof[m]) for m in FAMILIES] + [R(oof_e)], axis=0))
    n_te = R(np.mean([R(fte[m]) for m in FAMILIES] + [R(te_e)], axis=0))
    _, t3 = top1_f1(y, b_oof)
    _, t4 = top1_f1(y, n_oof)
    print(f"  三族 OOF {t3}/3000 -> 四族 {t4}/3000（{t4 - t3:+d}）")
    swap_report(b_oof, n_oof, y=y, label="融合 OOF")

    sil = pd.read_parquet(P("silence_feat.parquet"),
                          columns=["card_no", "sil_days"])
    sil["card_no"] = sil["card_no"].astype(str)
    hard = te.merge(sil, on="card_no", how="left")["sil_days"].values >= 32
    s, b = n_te.copy(), b_te.copy()
    if hard.any():
        s[hard] = 1.0
        b[hard] = 1.0
    swap_report(b, s, y=None, label=f"融合 {TARGET}")
    sub = te[["card_no"]].copy()
    sub["score"] = s
    out = os.path.join(LAB_PRED, f"pred_{TARGET}_{TAG}_4fam.csv")
    sub.to_csv(out, index=False)
    print(f"  候选文件（未提交）: {out}")

print(f"\n全部完成 {time.time()-t0:.0f}s")
