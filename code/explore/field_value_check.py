# -*- coding: utf-8 -*-
"""字段价值诊断（诊断脚本，不属于 01~10 正式复现链路，可随时单独运行）

验证三个字段是否值得投入特征工程：
  1. occupation：已知高风险职业码（学生/自由职业/无业/不便分类）的覆盖率与欺诈率
  2. industry：缺失率与 label 的关系 + fold-safe 目标编码单变量 AUC（不依赖码值语义）
  3. smy_cd：fold-safe 目标编码（按卡内各摘要码出现次数加权）单变量 AUC
             + 高覆盖摘要码清单（只打印码值和覆盖卡数，不含语义，供你自行去内网
               数据项编号 104142 核对）

前置条件：已跑过 01_parse_raw.py，data/interim/ 下存在
  card.parquet / cst.parquet / txn.parquet / train.parquet

用法:
  PYTHONPATH=code python code/explore/field_value_check.py

耗时：smy_cd 部分需要扫描全量 txn（约 997 万行），预计 3~6 分钟，其余数秒内完成。
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P, FOLD_SEED, N_FOLDS

t0 = time.time()

# ---------------- 基础数据 ----------------
train = pl.read_parquet(P("train.parquet")).select(["card_no", "label"]).to_pandas()
train["card_no"] = train["card_no"].astype(str)
train["label"] = train["label"].astype(int)

card = pl.read_parquet(P("card.parquet"), columns=["card_no", "cst_id"]).to_pandas()
cst = pl.read_parquet(P("cst.parquet"),
                      columns=["cst_id", "occupation", "industry"]).to_pandas()
card["card_no"] = card["card_no"].astype(str)
static = card.merge(cst, on="cst_id", how="left")

df = train.merge(static[["card_no", "occupation", "industry"]],
                 on="card_no", how="left")
y = df["label"].values
base_rate = df["label"].mean()
print(f"训练集卡数: {len(df):,}  全局基准欺诈率: {base_rate:.4%}")


def fold_safe_target_encode_simple(values, y, n_folds=N_FOLDS, seed=FOLD_SEED):
    """单值类别列的 fold-safe 目标编码，返回 OOF 编码数组"""
    n = len(values)
    oof = np.zeros(n, dtype=np.float64)
    global_mean = y.mean()
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    idx_all = np.arange(n)
    for tr_idx, va_idx in skf.split(idx_all, y):
        stat = pd.Series(y[tr_idx]).groupby(values[tr_idx]).mean()
        oof[va_idx] = pd.Series(values[va_idx]).map(stat).fillna(global_mean).values
    return oof


# ==================== 1. occupation 高风险职业验证 ====================
print("\n" + "=" * 60)
print("【1】occupation 高风险职业验证")
RISK_OCC = ["80000", "89900", "80400", "80200", "80500"]
df["is_risk_occ"] = df["occupation"].isin(RISK_OCC).astype(int)
print("-- 高风险组 vs 其余 --")
print(df.groupby("is_risk_occ")["label"].agg(["count", "mean"]))
print("-- 5 个码值各自统计 --")
sub = df[df["occupation"].isin(RISK_OCC)]
print(sub.groupby("occupation")["label"].agg(["count", "mean"]))
print("occupation 缺失率:", df["occupation"].isna().mean())


# ==================== 2. industry 缺失率 x label ====================
print("\n" + "=" * 60)
print("【2】industry 缺失率 x label")
df["industry_missing"] = df["industry"].isna().astype(int)
print(df.groupby("label")["industry_missing"].mean())
print("industry 唯一值数(非空):", df["industry"].nunique())


# ==================== 3. industry fold-safe 目标编码 + 单变量 AUC ====================
print("\n" + "=" * 60)
print("【3】industry fold-safe 目标编码 + 单变量 AUC")
ind_vals = df["industry"].fillna("__MISSING__").values
ind_te = fold_safe_target_encode_simple(ind_vals, y)
auc_ind = roc_auc_score(y, ind_te)
print(f"industry 目标编码单变量 AUC: {auc_ind:.4f}")

print("-- 目标编码欺诈率 Top10（匿名化，仅排名，不打印真实码值）--")
stat_full = pd.DataFrame({"code": ind_vals, "label": y})
stat_full = stat_full.groupby("code")["label"].agg(["count", "mean"])
stat_full = stat_full[stat_full["count"] >= 100].sort_values("mean", ascending=False).head(10)
stat_full = stat_full.reset_index(drop=True)
stat_full.index = [f"ind_rank{i+1}" for i in range(len(stat_full))]
print(stat_full)


# ==================== 4/5. smy_cd 目标编码 + 高覆盖码值清单 ====================
print("\n" + "=" * 60)
print("【4】smy_cd 高覆盖码值清单（仅码值+覆盖卡数，供你自行核对数据项 104142 语义）")
txn = pl.read_parquet(P("txn.parquet"), columns=["card_no", "smy_cd"])
smy_card = (txn.group_by(["card_no", "smy_cd"]).len()
            .rename({"len": "n"}).to_pandas())
smy_card["card_no"] = smy_card["card_no"].astype(str)
del txn

cov = (smy_card.groupby("smy_cd")["card_no"].nunique()
       .sort_values(ascending=False))
print(f"smy_cd 唯一码值数: {len(cov)}")
print("覆盖卡数 Top80：")
print(cov.head(80))

total_txn = smy_card["n"].sum()
top95_codes = set(cov.head(95).index)  # 对齐现有 02+05 脚本已展开的规模
top95_txn = smy_card[smy_card["smy_cd"].isin(top95_codes)]["n"].sum()
print(f"现有 Top95 展开方案覆盖交易占比: {top95_txn/total_txn:.2%}"
      f"（剩余 {1-top95_txn/total_txn:.2%} 落在长尾、被现有特征丢弃）")

print("\n" + "=" * 60)
print("【5】smy_cd fold-safe 目标编码（按卡内各码值出现次数加权平均）+ 单变量 AUC")
cards_df = df[["card_no", "label"]].reset_index(drop=True)
card_pos = {c: i for i, c in enumerate(cards_df["card_no"].values)}
smy_idx = smy_card.merge(cards_df, on="card_no", how="inner")  # 只保留 train 内的卡
global_mean = y.mean()
oof = np.full(len(cards_df), global_mean, dtype=np.float64)

skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED)
row_idx = np.arange(len(cards_df))
for k, (tr_idx, va_idx) in enumerate(skf.split(row_idx, y)):
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
    print(f"  fold{k} 完成 {time.time()-t0:.0f}s")

auc_smy = roc_auc_score(y, oof)
print(f"smy_cd 目标编码(加权平均)单变量 AUC: {auc_smy:.4f}")

print(f"\n全部完成 {time.time()-t0:.0f}s")
