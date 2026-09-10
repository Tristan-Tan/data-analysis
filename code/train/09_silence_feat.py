# -*- coding: utf-8 -*-
"""09 silence 特征族（22 维）—— 本方案的核心创新

【设计动机】
数据窗口是「每卡末笔交易往前 30 自然日」，因此此前所有时序特征都按
**相对天数**（T-1/T-3/T-7）构造，以避免 train/test 时间错位。
但这同时丢掉了一个绝对量：

    sil_days = (全量末笔日期的最大值) - (该卡末笔日期)

它刻画的是「这张卡沉默了多久」。电诈卡的终态是：受害人最后一笔汇入到账后
流水戛然而止（疑似被止付/冻结），所以沉默天数显著高于正常卡。

【实测判别力】
  · 单变量 AUC = 0.7702（label1 均值 16.61 天 vs label0 7.92 天）
  · 与 features_v4 全部 299 维的最高相关仅 0.491
    -> 这是一块模型此前**完全无法表达**的信息
  · 与活跃度的乘性交互极强（笔数 Q5 × 沉默 15+ 天的格子欺诈率 57.7%）

【关键设计】
原始 sil_days 本身价值有限（模型内 gain 仅 260），价值几乎全部来自
**它与活跃度的交互**与**组内分位**：
  · sil_pct_in_ntxn_bin（笔数分箱内的沉默分位）gain 12776，全 401 维第 4
  · ntxn_x_sil（log 笔数 × 沉默天数）        gain 11781，全 401 维第 5
组内分位这类特征 GBDT 学不出来（树无法在叶子内做全局排序），因此必须显式构造。

⚠ 锚点：复赛必须先确认 testb 的末笔日期分布未平移（见说明文档 §三.1）。

输出: data/interim/silence_feat.parquet   耗时约 2 分钟
"""
import os
import sys
import time
import json

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P, INTERIM, SIL_COLS, SIL_HELPER

t0 = time.time()


def main():
    t = pl.read_parquet(P("txn.parquet"),
                        columns=["card_no", "tms", "accno_txn_sn", "acct_bal",
                                 "inflow_amt", "outflow_amt"])
    t = t.with_columns(
        pl.col("accno_txn_sn").cast(pl.Int64, strict=False).fill_null(0).alias("_sn"))

    # 每卡末笔时间，回连算相对天数
    last = t.group_by("card_no").agg(pl.col("tms").max().alias("last_tms"))
    t = t.join(last, on="card_no", how="left")
    t = t.with_columns(
        ((pl.col("last_tms") - pl.col("tms")).dt.total_seconds() / 86400.0)
        .alias("dback"))

    g = t.group_by("card_no").agg([
        pl.len().alias("n_txn"),
        pl.col("tms").max().alias("last_tms"),
        pl.col("tms").min().alias("first_tms"),
        pl.col("inflow_amt").sum().alias("in_sum"),
        pl.col("outflow_amt").sum().alias("out_sum"),
        (pl.col("dback") <= 1).sum().alias("n_t1"),
        (pl.col("dback") <= 3).sum().alias("n_t3"),
        (pl.col("dback") <= 7).sum().alias("n_t7"),
        # 末笔余额 / 末笔方向：按 (tms, accno_txn_sn) 定序取最后一条
        pl.col("acct_bal").sort_by(["tms", "_sn"]).last().alias("bal_last"),
        pl.col("inflow_amt").sort_by(["tms", "_sn"]).last().alias("_last_in"),
    ])
    del t

    g = g.with_columns([
        pl.col("last_tms").dt.date().alias("last_dt"),
        pl.col("first_tms").dt.date().alias("first_dt"),
        pl.col("last_tms").dt.hour().alias("last_hour"),
        pl.col("last_tms").dt.weekday().alias("last_wd"),
        (pl.col("_last_in") > 0).cast(pl.Int8).alias("last_is_in"),
    ])
    df = g.to_pandas()
    df["card_no"] = df["card_no"].astype(str)
    del g

    # ---- 锚点：全量末笔日期的最大值，写盘供复赛核对 ----
    anchor_dt = pd.Timestamp(df["last_dt"].max())
    anchor_ts = anchor_dt + pd.Timedelta(days=1)
    print(f"锚点 ANCHOR_DATE = {anchor_dt.date()}")
    with open(os.path.join(INTERIM, "silence_anchor.json"), "w") as f:
        json.dump({"anchor_date": str(anchor_dt.date()),
                   "note": "复赛必须沿用同一锚点；若 testb 末笔日整体平移，"
                           "改用 testb last_dt 的 P99.9 校准，并同步重算 train 侧"},
                  f, ensure_ascii=False, indent=2)

    ld, fd = pd.to_datetime(df["last_dt"]), pd.to_datetime(df["first_dt"])
    df["sil_days"] = (anchor_dt - ld).dt.days.astype(np.int32)
    df["sil_hours"] = ((anchor_ts - pd.to_datetime(df["last_tms"]))
                       .dt.total_seconds() / 3600.0).astype(np.float32)
    df["first_sil_days"] = (anchor_dt - fd).dt.days.astype(np.int32)
    df["span_days"] = (ld - fd).dt.days.astype(np.int32)
    # 数据抽取伪影：正常卡末笔必在窗口起点之后，故 sil>=32 的卡在 train 中 100% 为欺诈
    df["sil_beyond_norm"] = (df["sil_days"] >= 32).astype(np.int8)
    df["last_is_bizhour"] = df["last_hour"].between(9, 16).astype(np.int8)
    df["last_is_deepnight"] = (df["last_hour"] <= 2).astype(np.int8)

    # ---- 交互族：活跃度 × 沉默（价值主体）----
    s = df["sil_days"].astype(np.float32)
    n = df["n_txn"].astype(np.float32)
    ins = df["in_sum"].astype(np.float32)
    outs = df["out_sum"].astype(np.float32)
    bal = df["bal_last"].astype(np.float32)

    df["ntxn_x_sil"] = np.log1p(n) * s
    df["ntxn_div_sil"] = n / (s + 1.0)
    df["insum_x_sil"] = np.log1p(np.clip(ins, 0, None)) * s
    df["bal_x_sil"] = np.log1p(np.clip(bal, 0, None)) * s
    df["t1share_x_sil"] = (df["n_t1"] / n) * s
    df["t3share_x_sil"] = (df["n_t3"] / n) * s
    net = np.clip(ins - outs, 0, None)
    df["netflow_x_sil"] = np.log1p(net) * s          # 高余额滞留 × 长沉默
    df["retention_x_sil"] = (net / (ins + 1.0)) * s
    df["txnrate_x_sil"] = (n / (df["span_days"] + 1.0)) * s
    df["sil_div_span"] = s / (df["span_days"] + 1.0)

    # ---- 组内分位（label-free，GBDT 学不出来）----
    nb = pd.qcut(df["n_txn"], 20, labels=False, duplicates="drop")
    df["sil_pct_in_ntxn_bin"] = df.groupby(nb)["sil_days"].rank(pct=True).astype(np.float32)
    sb = np.minimum(df["sil_days"], 32)
    df["ntxn_pct_in_sil_bin"] = df.groupby(sb)["n_txn"].rank(pct=True).astype(np.float32)
    df["bal_pct_in_sil_bin"] = df.groupby(sb)["bal_last"].rank(pct=True).astype(np.float32)
    df["insum_pct_in_sil_bin"] = df.groupby(sb)["in_sum"].rank(pct=True).astype(np.float32)

    df = df[["card_no"] + SIL_COLS + SIL_HELPER].copy()
    for c in SIL_COLS + SIL_HELPER:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.to_parquet(P("silence_feat.parquet"), index=False)
    print(f"silence_feat: {df.shape}（新增 {len(SIL_COLS)} 维 + "
          f"{len(SIL_HELPER)} 辅助）  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
