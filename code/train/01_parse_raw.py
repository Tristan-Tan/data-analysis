# -*- coding: utf-8 -*-
"""01 原始 CSV -> parquet + 完整性/时间锚点检查

输入: data/train/*.csv, data/testA/testa.csv
输出: data/interim/{txn,card,cst,accno,train,testa,anchor}.parquet
耗时: 约 3 分钟
"""
import os, sys, time
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_DIR, TESTA_DIR, P

t0 = time.time()

# ---------- 1. txn ----------
str_cols = ["card_no", "cst_accno", "cntrprt_card_no", "cntrprt_name",
            "cntrprt_clear_type", "txn_channel_type", "txn_cgycd", "smy_cd"]
ov = {c: pl.Utf8 for c in str_cols}
ov.update({"accno_txn_sn": pl.Int64, "acct_bal": pl.Float64, "txn_amt": pl.Float64,
           "inflow_amt": pl.Float64, "outflow_amt": pl.Float64,
           "cntrprt_prvt_ind": pl.Float64, "rmrk_recharge_withdrawal_ind": pl.Float64,
           "rmrk_short_ind": pl.Float64, "rmrk_e_credit_ind": pl.Float64})

txn = pl.read_csv(os.path.join(TRAIN_DIR, "txn_info_train_testa.csv"),
                  schema_overrides=ov)
txn = txn.with_columns([
    pl.col("tms").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
    pl.col("stm_dt").str.to_date("%Y-%m-%d", strict=False),
])
txn.write_parquet(P("txn.parquet"))
print(f"txn: {txn.shape}  {time.time()-t0:.0f}s")

# ---------- 2. 其余表 ----------
def conv(path, name, ovv=None):
    df = pl.read_csv(path, schema_overrides=ovv or {})
    df.write_parquet(P(name + ".parquet"))
    return df

card = conv(os.path.join(TRAIN_DIR, "card_info_train_testa.csv"), "card",
            {"card_no": pl.Utf8, "prim_cst_accno": pl.Utf8, "cst_id": pl.Utf8,
             "crdisu_lvl1_insid": pl.Utf8, "crdisu_lvl2_insid": pl.Utf8})
cst = conv(os.path.join(TRAIN_DIR, "cst_info_train_testa.csv"), "cst",
           {"cst_id": pl.Utf8, "gender": pl.Utf8, "marital_status": pl.Utf8,
            "occupation": pl.Utf8, "industry": pl.Utf8})
accno = conv(os.path.join(TRAIN_DIR, "accno_info_train_testa.csv"), "accno",
             {"card_no": pl.Utf8, "cst_accno": pl.Utf8})

# ⚠ train.csv 的 label 是 "0.0"/"1.0" 浮点文本，直接按 Int8 读会得到 null
train = conv(os.path.join(TRAIN_DIR, "train.csv"), "train",
             {"card_no": pl.Utf8, "label": pl.Float64})
train = train.with_columns(pl.col("label").cast(pl.Int8))
train.write_parquet(P("train.parquet"))

testa = conv(os.path.join(TESTA_DIR, "testa.csv"), "testa", {"card_no": pl.Utf8})
print(f"train={train.shape} testa={testa.shape} 欺诈率={train['label'].mean():.6f}")

# ---------- 3. 覆盖性检查 ----------
txn_cards = txn.select(pl.col("card_no").unique())
print("txn 唯一卡数:", txn_cards.height)
print("train 无流水卡:", train.join(txn_cards, on="card_no", how="anti").height)
print("testa 无流水卡:", testa.join(txn_cards, on="card_no", how="anti").height)

# ---------- 4. 时间锚点 ----------
anchor = txn.group_by("card_no").agg([
    pl.col("stm_dt").max().alias("last_dt"),
    pl.col("stm_dt").min().alias("first_dt"),
    pl.len().alias("n_txn")])
anchor = anchor.with_columns(
    (pl.col("last_dt") - pl.col("first_dt")).dt.total_days().alias("span_days"))
mem = pl.concat([
    train.select("card_no", pl.col("label").cast(pl.Utf8).alias("grp")),
    testa.select("card_no", pl.lit("testa").alias("grp"))])
anchor = anchor.join(mem, on="card_no", how="left")
anchor.write_parquet(P("anchor.parquet"))
print("末笔日期范围:", anchor["last_dt"].min(), "~", anchor["last_dt"].max())
print(f"01 完成 {time.time()-t0:.0f}s")
