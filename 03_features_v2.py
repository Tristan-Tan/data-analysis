# -*- coding: utf-8 -*-
"""03 特征 v2（142 列）：尾部行为 / 对手网络 / 摘要 / 金额形态
输出: data/interim/features_v2.parquet   耗时约 2 分钟
"""
import os, sys, time
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P

t0 = time.time()
txn = pl.read_parquet(P("txn.parquet")).sort(["card_no", "tms", "accno_txn_sn"])
txn = txn.with_columns([
    pl.col("tms").dt.hour().alias("hour"),
    (pl.col("inflow_amt") > 0).alias("is_in"),
    (pl.col("outflow_amt") > 0).alias("is_out"),
    pl.col("stm_dt").max().over("card_no").alias("anchor_dt"),
    (pl.len().over("card_no") - 1 - pl.int_range(pl.len()).over("card_no"))
        .alias("rn_end")])          # 0 = 末笔
txn = txn.with_columns(
    (pl.col("anchor_dt") - pl.col("stm_dt")).dt.total_days().alias("days_before"))

blocks = []
# 1. 尾部：末笔 / 最后 5 笔 / 末日
blocks.append(txn.filter(pl.col("rn_end") == 0).select([
    "card_no",
    pl.col("is_out").cast(pl.Int8).alias("last_is_out"),
    pl.col("txn_amt").alias("last_amt"),
    pl.col("acct_bal").alias("last_bal2"),
    pl.col("hour").alias("last_hour"),
    pl.col("cntrprt_prvt_ind").alias("last_prvt")]))
blocks.append(txn.filter(pl.col("rn_end") < 5).group_by("card_no").agg([
    pl.col("is_out").mean().alias("last5_out_share"),
    pl.col("outflow_amt").sum().alias("last5_out_sum"),
    pl.col("inflow_amt").sum().alias("last5_in_sum"),
    pl.col("txn_amt").mean().alias("last5_amt_mean"),
    pl.col("acct_bal").min().alias("last5_bal_min")]))
blocks.append(txn.filter(pl.col("days_before") == 0).group_by("card_no").agg([
    pl.len().alias("d0_n"),
    pl.col("inflow_amt").sum().alias("d0_in_sum"),
    pl.col("outflow_amt").sum().alias("d0_out_sum"),
    pl.col("cntrprt_card_no").n_unique().alias("d0_n_cntrprt"),
    pl.col("hour").min().alias("d0_hour_min"),
    pl.col("hour").max().alias("d0_hour_max")]))

# 2. 单日峰值
blocks.append(txn.group_by(["card_no", "stm_dt"]).agg([
    pl.len().alias("dn"), pl.col("outflow_amt").sum().alias("dout"),
    pl.col("inflow_amt").sum().alias("din")])
    .group_by("card_no").agg([
        pl.col("dn").max().alias("daily_n_max"),
        pl.col("dout").max().alias("daily_out_max"),
        pl.col("din").max().alias("daily_in_max"),
        pl.col("dn").std().alias("daily_n_std")]))

# 3. 熵
def entropy_of(col, name):
    g = txn.group_by(["card_no", col]).len()
    g = g.with_columns((pl.col("len") / pl.col("len").sum().over("card_no")).alias("p"))
    return g.group_by("card_no").agg(
        (-(pl.col("p") * pl.col("p").log()).sum()).alias(name))
blocks.append(entropy_of("hour", "hour_entropy"))
blocks.append(entropy_of("txn_channel_type", "channel_entropy"))

# 4. 金额重复形态
amt = (txn.group_by(["card_no", "txn_amt"]).len().group_by("card_no").agg([
    pl.col("len").max().alias("amt_top1_cnt"),
    pl.col("len").sum().alias("_tot"),
    pl.len().alias("n_unique_amt")]))
blocks.append(amt.with_columns([
    (pl.col("amt_top1_cnt") / pl.col("_tot")).alias("amt_top1_share"),
    (pl.col("n_unique_amt") / pl.col("_tot")).alias("amt_unique_ratio")]).drop("_tot"))

# 5. smy_cd Top15 占比
top_smy = (txn["smy_cd"].value_counts().sort("count", descending=True)
           .head(15)["smy_cd"].to_list())
blocks.append(txn.group_by("card_no").agg(
    [(pl.col("smy_cd") == s).mean().alias(f"smy_{s}_share") for s in top_smy]))

# 6. 对手方网络度
pairs = (txn.filter(pl.col("cntrprt_card_no").is_not_null())
         .select(["card_no", "cntrprt_card_no"]).unique())
deg = pairs.group_by("cntrprt_card_no").len().rename({"len": "cp_degree"})
pairs = pairs.join(deg, on="cntrprt_card_no", how="left")
cards_flag = txn.select(pl.col("card_no").unique()).with_columns(pl.lit(1).alias("is_sample"))
pairs = pairs.join(cards_flag.rename({"card_no": "cntrprt_card_no"}),
                   on="cntrprt_card_no", how="left")
blocks.append(pairs.group_by("card_no").agg([
    pl.col("cp_degree").max().alias("cpdeg_max"),
    pl.col("cp_degree").mean().alias("cpdeg_mean"),
    (pl.col("cp_degree") >= 5).mean().alias("cpdeg_hub_share"),
    pl.col("is_sample").fill_null(0).mean().alias("cp_is_sample_share")]))
print("对手账号也是样本卡的 pair 占比:",
      round(pairs["is_sample"].fill_null(0).mean(), 4))
del txn, pairs

feat = pl.read_parquet(P("features_v1.parquet"))
for b in blocks:
    feat = feat.join(b, on="card_no", how="left")
feat = feat.with_columns([
    (pl.col("in_sum_w1") / (pl.col("in_sum") + 1)).alias("in_w1_concentration"),
    (pl.col("d0_out_sum") / (pl.col("d0_in_sum") + 1)).alias("d0_out_in_ratio"),
    (pl.col("last5_out_sum") / (pl.col("in_sum") + 1)).alias("last5_drain_ratio")])
feat.write_parquet(P("features_v2.parquet"))
print(f"features_v2: {feat.shape}  {time.time()-t0:.0f}s")
