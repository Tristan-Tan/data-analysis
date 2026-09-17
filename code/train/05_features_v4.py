# -*- coding: utf-8 -*-
"""05 特征 v4（300 列，含 card_no）：smy/渠道全量展开 + 末段小时级时间窗
输出: data/interim/features_v4.parquet   耗时约 3 分钟
"""
import os, sys, time
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P

t0 = time.time()
txn = pl.read_parquet(P("txn.parquet")).sort(["card_no", "tms", "accno_txn_sn"])
txn = txn.with_columns([
    (pl.col("inflow_amt") > 0).alias("is_in"),
    (pl.col("outflow_amt") > 0).alias("is_out"),
    pl.col("tms").max().over("card_no").alias("anchor_tms"),
    (pl.len().over("card_no") - 1 - pl.int_range(pl.len()).over("card_no"))
        .alias("rn_end")])
txn = txn.with_columns(
    (pl.col("anchor_tms") - pl.col("tms")).dt.total_seconds().alias("sec_before"))

blocks = []
# 1. smy_cd 全量展开（覆盖 >=300 卡，去掉 v2 已展开的 Top15）
OLD_SMY = ['0295','0457','0414','1151','1140','0544','8115','0968',
           '0240','7177','8363','0526','0336','0239','2072']
cov = (txn.group_by("smy_cd").agg(pl.col("card_no").n_unique().alias("nc"))
       .filter(pl.col("nc") >= 300).sort("nc", descending=True))
smy_codes = [s for s in cov["smy_cd"].to_list()
             if s is not None and s not in OLD_SMY][:80]
print(f"smy_cd 新增展开 {len(smy_codes)} 个码")
blocks.append(txn.group_by("card_no").agg(
    [(pl.col("smy_cd") == s).mean().alias(f"smy2_{s}") for s in smy_codes]))

# 2. 渠道全量展开
OLD_CH = ["0097","0056","0025","0006","0053","0003","0018",
          "9999","0001","0052","8100","0116","0029","0038"]
cov = (txn.group_by("txn_channel_type").agg(pl.col("card_no").n_unique().alias("nc"))
       .filter(pl.col("nc") >= 300).sort("nc", descending=True))
ch_codes = [s for s in cov["txn_channel_type"].to_list()
            if s is not None and s not in OLD_CH][:40]
print(f"渠道新增展开 {len(ch_codes)} 个码")
blocks.append(txn.group_by("card_no").agg(
    [(pl.col("txn_channel_type") == s).mean().alias(f"ch2_{s}") for s in ch_codes]))

# 3. 末段小时级时间窗
for tag, sec in [("h1", 3600), ("h6", 21600), ("h24", 86400)]:
    f = pl.col("sec_before") < sec
    blocks.append(txn.group_by("card_no").agg([
        f.sum().alias(f"{tag}_n"),
        pl.col("outflow_amt").filter(f).sum().alias(f"{tag}_out_sum"),
        pl.col("inflow_amt").filter(f).sum().alias(f"{tag}_in_sum"),
        pl.col("cntrprt_card_no").filter(f).n_unique().alias(f"{tag}_n_cp"),
        pl.col("cntrprt_prvt_ind").filter(f).mean().alias(f"{tag}_prvt"),
        (pl.col("cntrprt_clear_type").filter(f) == "interbank_txn").mean()
            .alias(f"{tag}_interbank")]))

# 4. last3 / last20
for tag, n in [("last3", 3), ("last20", 20)]:
    blocks.append(txn.filter(pl.col("rn_end") < n).group_by("card_no").agg([
        pl.col("is_out").mean().alias(f"{tag}_out_share"),
        pl.col("outflow_amt").sum().alias(f"{tag}_out_sum"),
        pl.col("cntrprt_card_no").n_unique().alias(f"{tag}_n_cp"),
        pl.col("cntrprt_prvt_ind").mean().alias(f"{tag}_prvt"),
        pl.col("txn_amt").mean().alias(f"{tag}_amt_mean")]))

# 5. 入金对手集中度 + 星期
inc = (txn.filter(pl.col("is_in") & pl.col("cntrprt_card_no").is_not_null())
       .group_by(["card_no", "cntrprt_card_no"])
       .agg(pl.col("inflow_amt").sum().alias("s"))
       .group_by("card_no").agg([pl.col("s").max().alias("_inmax"),
                                 pl.col("s").sum().alias("_intot"),
                                 pl.len().alias("n_in_cp")]))
blocks.append(inc.with_columns(
    (pl.col("_inmax") / (pl.col("_intot") + 1)).alias("in_cp_top1_share")
).drop(["_inmax", "_intot"]))
blocks.append(txn.group_by("card_no").agg([
    (pl.col("tms").dt.weekday() >= 6).mean().alias("weekend_share"),
    pl.col("anchor_tms").first().dt.weekday().alias("anchor_weekday"),
    pl.col("anchor_tms").first().dt.hour().alias("anchor_hour")]))
del txn

feat = pl.read_parquet(P("features_v3.parquet"))
for b in blocks:
    feat = feat.join(b, on="card_no", how="left")
feat = feat.with_columns([
    (pl.col("h24_out_sum") / (pl.col("out_sum") + 1)).alias("h24_out_concentration"),
    (pl.col("h1_n") / (pl.col("n_txn") + 1)).alias("h1_txn_share")])
feat.write_parquet(P("features_v4.parquet"))
print(f"features_v4: {feat.shape}  {time.time()-t0:.0f}s")
