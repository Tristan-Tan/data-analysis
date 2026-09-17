# -*- coding: utf-8 -*-
"""04 特征 v3（174 列）：末段序列 / 末次入金后行为 / 分散转出 / 新对手
输出: data/interim/features_v3.parquet   耗时约 3 分钟
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
    pl.col("tms").max().over("card_no").alias("anchor_tms"),
    (pl.len().over("card_no") - 1 - pl.int_range(pl.len()).over("card_no"))
        .alias("rn_end")])
txn = txn.with_columns(
    (pl.col("anchor_dt") - pl.col("stm_dt")).dt.total_days().alias("days_before"))

blocks = []
# 1. 末次入金位置与其后的行为
last_in = txn.filter(pl.col("is_in")).group_by("card_no").agg([
    pl.col("rn_end").min().alias("trailing_outs"),
    pl.col("tms").max().alias("last_in_tms"),
    pl.col("inflow_amt").last().alias("last_in_amt"),
    pl.col("days_before").min().alias("last_in_days_before")])
anchor_tms = txn.group_by("card_no").agg(pl.col("anchor_tms").first())
last_in = last_in.join(anchor_tms, on="card_no", how="left").with_columns(
    (pl.col("anchor_tms") - pl.col("last_in_tms")).dt.total_seconds()
    .alias("sec_lastin_to_end")).drop(["last_in_tms", "anchor_tms"])
blocks.append(last_in)

t2 = txn.join(last_in.select(["card_no", "trailing_outs"]), on="card_no", how="left")
after_in = (t2.filter(pl.col("rn_end") < pl.col("trailing_outs"))
            .group_by("card_no").agg([
                pl.col("outflow_amt").sum().alias("afterin_out_sum"),
                pl.col("cntrprt_card_no").n_unique().alias("afterin_n_cntrprt"),
                pl.col("tms").min().alias("_t0"), pl.col("tms").max().alias("_t1")]))
blocks.append(after_in.with_columns(
    (pl.col("_t1") - pl.col("_t0")).dt.total_seconds().alias("afterin_span_sec")
).drop(["_t0", "_t1"]))
del t2

# 2. 末段序列形态
blocks.append(txn.filter(pl.col("rn_end") < 10).group_by("card_no").agg([
    pl.col("is_out").mean().alias("last10_out_share"),
    pl.col("outflow_amt").sum().alias("last10_out_sum"),
    pl.col("cntrprt_card_no").n_unique().alias("last10_n_cntrprt"),
    pl.col("cntrprt_prvt_ind").mean().alias("last10_prvt_share"),
    (pl.col("cntrprt_clear_type") == "interbank_txn").mean().alias("last10_interbank"),
    pl.col("txn_amt").max().alias("last10_amt_max"),
    pl.col("hour").mean().alias("last10_hour_mean")]))
blocks.append(txn.with_columns(
    (pl.col("is_out") & pl.col("is_in").shift(1).over("card_no")).alias("io_flip")
).group_by("card_no").agg([pl.col("io_flip").sum().alias("n_io_flips")]))

lastrow = txn.filter(pl.col("rn_end") == 0)
cp_first = (txn.filter(pl.col("cntrprt_card_no").is_not_null())
            .group_by(["card_no", "cntrprt_card_no"])
            .agg(pl.col("days_before").max().alias("cp_first_days")))
lastrow = lastrow.join(cp_first, on=["card_no", "cntrprt_card_no"], how="left")
blocks.append(lastrow.select([
    "card_no",
    (pl.col("cntrprt_clear_type") == "interbank_txn").cast(pl.Int8).alias("last_interbank"),
    (pl.col("cp_first_days") == 0).cast(pl.Int8).alias("last_cp_is_new"),
    (pl.col("txn_channel_type") == "0097").cast(pl.Int8).alias("last_ch_0097"),
    (pl.col("txn_channel_type") == "0056").cast(pl.Int8).alias("last_ch_0056"),
    (pl.col("txn_channel_type") == "0006").cast(pl.Int8).alias("last_ch_0006"),
    (pl.col("txn_channel_type") == "0053").cast(pl.Int8).alias("last_ch_0053")]))

# 3. 分散转出 fan-out
blocks.append(txn.filter(pl.col("is_out") & pl.col("cntrprt_card_no").is_not_null())
              .group_by(["card_no", "stm_dt"])
              .agg(pl.col("cntrprt_card_no").n_unique().alias("day_out_cps"))
              .group_by("card_no").agg([
                  pl.col("day_out_cps").max().alias("fanout_day_max"),
                  pl.col("day_out_cps").mean().alias("fanout_day_mean")]))
blocks.append(txn.filter(pl.col("is_out")).group_by(["card_no", "stm_dt", "hour"]).len()
              .group_by("card_no").agg(pl.col("len").max().alias("hourly_out_max")))

# 4. 末日新对手出金
blocks.append(txn.filter((pl.col("days_before") == 0) & pl.col("is_out")
                         & pl.col("cntrprt_card_no").is_not_null())
              .join(cp_first, on=["card_no", "cntrprt_card_no"], how="left")
              .group_by("card_no").agg([
                  (pl.col("cp_first_days") == 0).mean().alias("d0_newcp_share"),
                  pl.col("outflow_amt").filter(pl.col("cp_first_days") == 0).sum()
                      .alias("d0_newcp_out_sum")]))

# 5. 余额轨迹
blocks.append(txn.group_by("card_no").agg([
    pl.col("acct_bal").first().alias("bal_first"),
    pl.col("acct_bal").filter(pl.col("rn_end") == 1).first().alias("bal_prev1")]))
del txn

feat = pl.read_parquet(P("features_v2.parquet"))
for b in blocks:
    feat = feat.join(b, on="card_no", how="left")
feat = feat.with_columns([
    (pl.col("bal_last") - pl.col("bal_first")).alias("bal_delta"),
    (pl.col("last_in_amt") / (pl.col("bal_max") + 1)).alias("lastin_vs_balmax"),
    (pl.col("afterin_out_sum") / (pl.col("last_in_amt") + 1)).alias("afterin_drain"),
    (pl.col("trailing_outs") / (pl.col("n_txn") + 1)).alias("trailing_out_ratio")])
feat.write_parquet(P("features_v3.parquet"))
print(f"features_v3: {feat.shape}  {time.time()-t0:.0f}s")
