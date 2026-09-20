# -*- coding: utf-8 -*-
"""train vs testB 全特征分布漂移诊断（对抗验证）

B 榜首次提交 663/754（0.87931），比 A 榜最好成绩 691/754 少命中 28 张，
远超正常复现噪声（±3 张）。本脚本用对抗验证定位原因：把 train 标 0、
testB 标 1，用全部 402 维训练一个分类器区分两侧，AUC 越高说明两侧分布
差异越大；再看哪些维度贡献了这个差异。

方法论同审查材料 §一.5（exp5 标签密度修正时用的对抗验证，0.9695→0.4989）。

【怎么看结果】
  · 对抗 AUC 接近 0.5     -> 两侧同分布，漂移不是掉分主因，要往别处查
  · 对抗 AUC 明显高于 0.5 -> 看 Top 维度是"合理偏移"还是"有害偏移"：
      合理（预期内，不必处理）：anchor_weekday / anchor_hour / last_wd /
        last_hour 等绝对日历派生量——testB 本来就是另一个月的数据
      有害（需要处理）：card_age_d / acc_age_*_d（testB 末笔晚一个月，
        卡龄被系统性拉大约 30 天）、exp5_* 暴露编码、smy_te——这些是
        模型学到的风险阈值赖以成立的量，偏移会直接让阈值失准

输出末尾单列 smy_te 专项对比（分布 + 摘要码覆盖率）。

用法: PYTHONPATH=code python code/explore/testb_drift_check.py
耗时: 约 5~10 分钟（单个 LightGBM + 少量统计）
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P, N_FOLDS, LGB_BASE
import assemble

t0 = time.time()

# ---------------- 两侧特征矩阵 ----------------
tr, Xtr, y_true, cols = assemble.build_train()
print(f"train 矩阵 {Xtr.shape}   {time.time()-t0:.0f}s")

te, Xte_list = assemble.build_test_folds(N_FOLDS, cols)
Xte = Xte_list[0]          # 折间只有 exp5/smy_te 编码不同，取第 0 折做诊断即可
print(f"testB 矩阵 {Xte.shape}（取 fold0）   {time.time()-t0:.0f}s")

# ---------------- 对抗验证 ----------------
X = pd.concat([Xtr, Xte], ignore_index=True)
y_adv = np.r_[np.zeros(len(Xtr), dtype=np.int8), np.ones(len(Xte), dtype=np.int8)]

oof = np.zeros(len(X))
gain = np.zeros(len(cols))
folds = StratifiedKFold(5, shuffle=True, random_state=42).split(X, y_adv)
for k, (ti, vi) in enumerate(folds):
    m = lgb.train(dict(LGB_BASE, seed=42),
                  lgb.Dataset(X.iloc[ti], y_adv[ti]), 1000,
                  valid_sets=[lgb.Dataset(X.iloc[vi], y_adv[vi])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    oof[vi] = m.predict(X.iloc[vi], num_iteration=m.best_iteration)
    gain += m.feature_importance("gain")
    print(f"  fold{k} best_iter={m.best_iteration}   {time.time()-t0:.0f}s")

auc = roc_auc_score(y_adv, oof)
print("\n" + "=" * 70)
print(f"【对抗 AUC】train vs testB = {auc:.4f}   (0.5 = 两侧同分布)")

# ---------------- 最可分辨的维度 + 两侧取值对比 ----------------
imp = pd.DataFrame({"feature": cols, "gain": gain}).sort_values(
    "gain", ascending=False).head(25).reset_index(drop=True)
imp["train_mean"] = [Xtr[f].mean() for f in imp["feature"]]
imp["testb_mean"] = [Xte[f].mean() for f in imp["feature"]]
imp["train_med"] = [Xtr[f].median() for f in imp["feature"]]
imp["testb_med"] = [Xte[f].median() for f in imp["feature"]]
imp["gain_share"] = imp["gain"] / gain.sum()

print("\n【最可分辨的 25 个维度】(gain_share = 占对抗模型总 gain 比例)")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(imp[["feature", "gain_share", "train_mean", "testb_mean",
           "train_med", "testb_med"]].to_string(index=False))

# ---------------- smy_te 专项 ----------------
print("\n" + "=" * 70)
print("【smy_te 专项】")
if "smy_te" in cols:
    q = [0.01, 0.25, 0.5, 0.75, 0.99]
    print("  train  分位:", np.round(Xtr["smy_te"].quantile(q).values, 6))
    print("  testB  分位:", np.round(Xte["smy_te"].quantile(q).values, 6))
    print(f"  train  mean={Xtr['smy_te'].mean():.6f} std={Xtr['smy_te'].std():.6f}")
    print(f"  testB  mean={Xte['smy_te'].mean():.6f} std={Xte['smy_te'].std():.6f}")
    print(f"  smy_te 在对抗模型里的 gain 排名: "
          f"{list(imp['feature']).index('smy_te') + 1 if 'smy_te' in list(imp['feature']) else '25名开外'}")
else:
    print("  当前矩阵不含 smy_te 列")

# 摘要码覆盖率：testB 的交易里有多少落在 train 见过的 smy_cd 上
txn = pl.read_parquet(P("txn.parquet"), columns=["card_no", "smy_cd"])
tr_cards = set(pd.read_parquet(P("train.parquet"))["card_no"].astype(str))
tb_cards = set(pd.read_parquet(P("testb.parquet"))["card_no"].astype(str))
sc = txn.to_pandas()
sc["card_no"] = sc["card_no"].astype(str)
codes_tr = set(sc.loc[sc["card_no"].isin(tr_cards), "smy_cd"].dropna().unique())
tb = sc[sc["card_no"].isin(tb_cards)]
cov = tb["smy_cd"].isin(codes_tr).mean()
print(f"\n  testB 交易落在 train 已见摘要码上的比例: {cov:.4%}"
      f"（越低说明越多 testB 交易拿不到编码、被 global_mean 兜底）")
unseen = tb.loc[~tb["smy_cd"].isin(codes_tr), "smy_cd"].value_counts().head(10)
if len(unseen):
    print("  testB 独有摘要码 Top10（train 里没出现过）:")
    print(unseen.to_string())

print(f"\n全部完成 {time.time()-t0:.0f}s")
