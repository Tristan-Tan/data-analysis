# -*- coding: utf-8 -*-
"""02 卡级基础特征 v1（94 列）
窗口聚合 / 快进快出 / 余额 / 对手统计 / 时间模式 / 金额形态 / 静态表
输出: data/interim/features_v1.parquet   耗时约 2 分钟
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
    pl.col("stm_dt").max().over("card_no").alias("anchor_dt")])
txn = txn.with_columns(
    (pl.col("anchor_dt") - pl.col("stm_dt")).dt.total_days().alias("days_before"))

# 快进快出：每笔出金距最近一笔入金的秒数
txn = txn.with_columns(
    pl.when(pl.col("is_in")).then(pl.col("tms")).otherwise(None).alias("in_tms"))
txn = txn.with_columns(
    pl.col("in_tms").forward_fill().over("card_no").alias("last_in_tms"))
txn = txn.with_columns(
    pl.when(pl.col("is_out"))
      .then((pl.col("tms") - pl.col("last_in_tms")).dt.total_seconds())
      .otherwise(None).alias("out_after_in_sec"))
txn = txn.with_columns(
    (pl.col("tms") - pl.col("tms").shift(1).over("card_no"))
    .dt.total_seconds().alias("gap_sec"))

CH = ["0097", "0056", "0025", "0006", "0053", "0003", "0018",
      "9999", "0001", "0052", "8100", "0116", "0029", "0038"]
aggs = [
    pl.len().alias("n_txn"),
    pl.col("stm_dt").n_unique().alias("n_days"),
    pl.col("days_before").max().alias("span_days"),
    pl.col("cst_accno").n_unique().alias("n_accno_used"),
    pl.col("is_in").sum().alias("n_in"),
    pl.col("inflow_amt").sum().alias("in_sum"),
    pl.col("inflow_amt").filter(pl.col("is_in")).mean().alias("in_mean"),
    pl.col("inflow_amt").max().alias("in_max"),
    pl.col("inflow_amt").filter(pl.col("is_in")).std().alias("in_std"),
    pl.col("is_out").sum().alias("n_out"),
    pl.col("outflow_amt").sum().alias("out_sum"),
    pl.col("outflow_amt").filter(pl.col("is_out")).mean().alias("out_mean"),
    pl.col("outflow_amt").max().alias("out_max"),
    pl.col("outflow_amt").filter(pl.col("is_out")).std().alias("out_std"),
    pl.col("acct_bal").last().alias("bal_last"),
    pl.col("acct_bal").min().alias("bal_min"),
    pl.col("acct_bal").max().alias("bal_max"),
    pl.col("acct_bal").mean().alias("bal_mean"),
    pl.col("acct_bal").std().alias("bal_std"),
    pl.col("cntrprt_card_no").n_unique().alias("n_cntrprt"),
    pl.col("cntrprt_prvt_ind").mean().alias("prvt_share"),
    (pl.col("cntrprt_clear_type") == "interbank_txn").mean().alias("interbank_share"),
    (pl.col("cntrprt_clear_type") == "on_us_bank_txn").mean().alias("onus_share"),
    (pl.col("txn_cgycd") == "01").mean().alias("cash_share"),
    pl.col("smy_cd").n_unique().alias("n_smy"),
    pl.col("txn_channel_type").n_unique().alias("n_channel"),
    pl.col("rmrk_recharge_withdrawal_ind").mean().alias("rmrk_rw_share"),
    pl.col("rmrk_short_ind").mean().alias("rmrk_short_share"),
    pl.col("rmrk_e_credit_ind").mean().alias("rmrk_ec_share"),
    (pl.col("hour") < 6).mean().alias("night_share"),
    pl.col("hour").n_unique().alias("n_hours"),
    pl.col("gap_sec").mean().alias("gap_mean"),
    pl.col("gap_sec").median().alias("gap_median"),
    pl.col("gap_sec").min().alias("gap_min"),
    (pl.col("gap_sec") < 60).sum().alias("n_burst60"),
    pl.col("out_after_in_sec").min().alias("fio_min"),
    pl.col("out_after_in_sec").median().alias("fio_median"),
    (pl.col("out_after_in_sec") < 3600).sum().alias("n_fastout_1h"),
    (pl.col("out_after_in_sec") < 86400).sum().alias("n_fastout_1d"),
    ((pl.col("txn_amt") % 100 == 0) & (pl.col("txn_amt") > 0)).mean().alias("round100_share"),
    ((pl.col("txn_amt") % 1000 == 0) & (pl.col("txn_amt") > 0)).mean().alias("round1000_share"),
    (pl.col("txn_amt") >= 10000).sum().alias("n_big1w"),
    (pl.col("txn_amt") >= 50000).sum().alias("n_big5w"),
]
for w in [1, 3, 7]:
    f = pl.col("days_before") < w
    aggs += [f.sum().alias(f"n_txn_w{w}"),
             pl.col("inflow_amt").filter(f).sum().alias(f"in_sum_w{w}"),
             pl.col("outflow_amt").filter(f).sum().alias(f"out_sum_w{w}")]
for ch in CH:
    aggs.append((pl.col("txn_channel_type") == ch).mean().alias(f"ch{ch}_share"))

feat = txn.group_by("card_no").agg(aggs)

cp = (txn.filter(pl.col("cntrprt_card_no").is_not_null())
      .group_by(["card_no", "cntrprt_card_no"]).len()
      .group_by("card_no").agg([pl.col("len").max().alias("cp_top1_cnt"),
                                pl.col("len").sum().alias("cp_cnt_total")]))
cp = cp.with_columns((pl.col("cp_top1_cnt") / pl.col("cp_cnt_total")).alias("cp_top1_share"))
feat = feat.join(cp, on="card_no", how="left")
anchor_dt = txn.group_by("card_no").agg(pl.col("anchor_dt").first())
del txn

feat = feat.with_columns([
    (pl.col("out_sum") / (pl.col("in_sum") + pl.col("out_sum") + 1)).alias("out_ratio"),
    (pl.col("in_sum") - pl.col("out_sum")).alias("net_flow"),
    (pl.col("n_txn") / pl.col("n_days")).alias("txn_per_day"),
    (pl.col("bal_last") / (pl.col("bal_max") + 1)).alias("bal_drain"),
    (pl.col("n_cntrprt") / pl.col("n_txn")).alias("cntrprt_per_txn"),
    (pl.col("n_txn_w7") / pl.col("n_txn")).alias("w7_txn_share"),
    (pl.col("n_txn_w1") / pl.col("n_txn")).alias("w1_txn_share"),
    (pl.col("out_sum") / (pl.col("bal_max") + 1)).alias("turnover_rate")])

card = pl.read_parquet(P("card.parquet"))
cst = pl.read_parquet(P("cst.parquet"))
accno = pl.read_parquet(P("accno.parquet"))
card = card.with_columns(pl.col("card_create_dt").str.to_date("%Y-%m-%d", strict=False))
accno = accno.with_columns(pl.col("open_accno_dt").str.to_date("%Y-%m-%d", strict=False))
card = card.join(card.group_by("cst_id").len().rename({"len": "cst_n_cards"}),
                 on="cst_id", how="left")
acc_agg = accno.group_by("card_no").agg([
    pl.len().alias("n_accounts"),
    pl.col("open_accno_dt").min().alias("acc_open_min"),
    pl.col("open_accno_dt").max().alias("acc_open_max")])
static = (card.join(cst, on="cst_id", how="left")
          .join(acc_agg, on="card_no", how="left")
          .join(anchor_dt, on="card_no", how="left"))
static = static.with_columns([
    (pl.col("anchor_dt") - pl.col("card_create_dt")).dt.total_days().alias("card_age_d"),
    (pl.col("anchor_dt") - pl.col("acc_open_min")).dt.total_days().alias("acc_age_max_d"),
    (pl.col("anchor_dt") - pl.col("acc_open_max")).dt.total_days().alias("acc_age_min_d")])
for c in ["crdisu_lvl1_insid", "crdisu_lvl2_insid", "gender", "marital_status",
          "occupation", "industry"]:
    vc = static.group_by(c).len().rename({"len": f"{c}_freq"})
    static = static.join(vc, on=c, how="left").drop(c)
static = static.drop(["prim_cst_accno", "cst_id", "card_create_dt",
                      "acc_open_min", "acc_open_max", "anchor_dt"])
feat = feat.join(static, on="card_no", how="left")
feat.write_parquet(P("features_v1.parquet"))
print(f"features_v1: {feat.shape}  {time.time()-t0:.0f}s")
