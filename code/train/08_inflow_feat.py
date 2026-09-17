# -*- coding: utf-8 -*-
"""08 入金侧 label-free 特征（19 维）：发送方扇入度 + 入金细粒度
输出: data/interim/inflow_feat.parquet   耗时约 2 分钟
"""
import os, sys, time
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P

t0 = time.time()
txn = pl.read_parquet(P("txn.parquet")).sort(["card_no", "tms", "accno_txn_sn"])
txn = txn.with_columns([
    (pl.col("inflow_amt") > 0).alias("is_in"),
    pl.col("tms").max().over("card_no").alias("anchor_tms"),
    pl.col("tms").dt.hour().alias("hour")])
ins = txn.filter(pl.col("is_in"))

# 发送方扇入度（账号级 fa_ / 名称级 fn_）
fe = None
for keycol, tag in [("cntrprt_card_no", "fa"), ("cntrprt_name", "fn")]:
    ie = (ins.filter(pl.col(keycol).is_not_null())
          .group_by(["card_no", keycol])
          .agg([pl.len().alias("n"), pl.col("inflow_amt").sum().alias("amt")]))
    deg = ie.group_by(keycol).agg(pl.col("card_no").n_unique().alias("deg"))
    ie = ie.join(deg, on=keycol, how="left")
    g = ie.group_by("card_no").agg([
        pl.col("deg").max().alias(f"{tag}_deg_max"),
        pl.col("deg").mean().alias(f"{tag}_deg_mean"),
        (pl.col("amt").filter(pl.col("deg") >= 2).sum()
         / (pl.col("amt").sum() + 1)).alias(f"{tag}_amt_deg2_share"),
        (pl.col("amt").filter(pl.col("deg") >= 5).sum()
         / (pl.col("amt").sum() + 1)).alias(f"{tag}_amt_deg5_share"),
        (pl.col("deg") >= 2).sum().alias(f"{tag}_n_senders_deg2"),
        (pl.col("n") == 1).mean().alias(f"{tag}_oneoff_share")])
    fe = g if fe is None else fe.join(g, on="card_no", how="full", coalesce=True)

g2 = ins.group_by("card_no").agg([
    pl.col("cntrprt_card_no").filter(pl.col("cntrprt_prvt_ind") == 1)
        .n_unique().alias("n_in_prvt_senders"),
    pl.col("cntrprt_card_no").n_unique().alias("n_in_senders"),
    (pl.col("hour") < 6).mean().alias("in_night_share"),
    pl.col("inflow_amt").max().alias("in_max2"),
    ((pl.col("inflow_amt") % 1000 == 0)).mean().alias("in_round1k_share"),
    (pl.col("cntrprt_clear_type") == "interbank_txn").mean().alias("in_interbank_share"),
    pl.col("tms").sort_by(pl.col("inflow_amt")).last().alias("_maxin_tms"),
    pl.col("anchor_tms").first().alias("_anchor")])
g2 = g2.with_columns(
    (pl.col("_anchor") - pl.col("_maxin_tms")).dt.total_seconds()
    .alias("sec_maxin_to_end")).drop(["_maxin_tms", "_anchor"])
fe = fe.join(g2, on="card_no", how="full", coalesce=True)
fe.write_parquet(P("inflow_feat.parquet"))
print(f"inflow_feat: {fe.shape}  {time.time()-t0:.0f}s")
