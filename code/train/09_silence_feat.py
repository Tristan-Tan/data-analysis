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

【B 榜实测更新】testB 末笔日期整体比 train/testA 晚了约一个月（testB 是独立
于 A 榜快照的、更晚抽取的一批数据，不是几天量级的漂移）。实测证实：若强行
用"全量最大值"做单一锚点，train 侧 sil_days 会被整体拉高约 30 天（label0
均值 7.92→37.92，sil≥30 占比从 1.78%→100%），特征完全失去区分度。

正确处理：**train+testA 与 testB 分别使用各自批次内部的末笔日期最大值作为
锚点**（而不是审查材料 §三.1 原计划里"统一锚点、同步重算训练侧"的方案——
该方案只适用于几天量级的漂移，遇到整月量级的批次间隔并不成立）。train 侧
锚点因此保持 A 榜原值不变，sil_days 数值与 A 榜完全一致；testB 侧单独用
自己的末笔日期最大值定锚，语义上仍是"该卡距其所属批次快照时间点的沉默
天数"，两批次可比。

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

    # ---- 锚点：train+testA 与 testB 分属两个独立快照批次，各自用批次内部
    #      末笔日期最大值定锚（不可强行统一，见上方模块说明）----
    testb_path = P("testb.parquet")
    has_testb = os.path.exists(testb_path)
    if has_testb:
        testb_cards = set(pd.read_parquet(testb_path)["card_no"].astype(str))
        df["is_testb"] = df["card_no"].isin(testb_cards)
    else:
        df["is_testb"] = False

    anchor_a = pd.Timestamp(df.loc[~df["is_testb"], "last_dt"].max())
    if has_testb and df["is_testb"].any():
        anchor_b = pd.Timestamp(df.loc[df["is_testb"], "last_dt"].max())
    else:
        anchor_b = anchor_a
    print(f"锚点 ANCHOR_A(train+testA) = {anchor_a.date()}  "
          f"ANCHOR_B(testB) = {anchor_b.date()}")
    with open(os.path.join(INTERIM, "silence_anchor.json"), "w") as f:
        json.dump({"anchor_date_a": str(anchor_a.date()),
                   "anchor_date_b": str(anchor_b.date()) if has_testb else None,
                   "note": "train+testA 与 testB 分属不同快照批次，各自用"
                           "批次内部末笔日期最大值定锚，不做跨批次统一"},
                  f, ensure_ascii=False, indent=2)

    anchor_dt_row = pd.Series(np.where(df["is_testb"], anchor_b, anchor_a),
                              index=df.index)
    anchor_dt_row = pd.to_datetime(anchor_dt_row)
    anchor_ts_row = anchor_dt_row + pd.Timedelta(days=1)

    ld, fd = pd.to_datetime(df["last_dt"]), pd.to_datetime(df["first_dt"])
    df["sil_days"] = (anchor_dt_row - ld).dt.days.astype(np.int32)
    df["sil_hours"] = ((anchor_ts_row - pd.to_datetime(df["last_tms"]))
                       .dt.total_seconds() / 3600.0).astype(np.float32)
    df["first_sil_days"] = (anchor_dt_row - fd).dt.days.astype(np.int32)
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
