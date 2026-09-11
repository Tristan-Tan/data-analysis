# -*- coding: utf-8 -*-
"""01 原始 CSV -> parquet + 完整性/时间锚点检查

输入: data/train/*.csv, data/testA/testa.csv
     data/testB/{testb,txn_info_testb,card_info_testb,cst_info_testb,
                 accno_info_testb}.csv（复赛阶段，若存在则自动并入）
输出: data/interim/{txn,card,cst,accno,train,testa,testb,anchor}.parquet

⚠ B 榜的 txn_info_testb.csv 是独立文件，不在 txn_info_train_testa.csv 里，
  必须并入同一份 txn.parquet 才能保证 cp_degree（对手网络度）、exp5 暴露编码
  这类跨卡图特征在 train / testB 之间口径一致（同 A 榜 train_testa 合并的
  设计意图）。card/cst/accno 同理并入，testB 卡号应与 train+testA 不重叠。
  B 榜数据不存在时（如提前跑通 A 榜复现）自动跳过，行为与之前完全一致。

耗时: 约 3 分钟（B 榜数据到手后视数据量小幅增加）
"""
import os, sys, time
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_DIR, TESTA_DIR, TESTB_DIR, P

t0 = time.time()
HAS_TESTB = os.path.exists(os.path.join(TESTB_DIR, "testb.csv"))
print(f"testB 数据{'已就绪，将合并处理' if HAS_TESTB else '尚未就绪，跳过（仅处理 train+testA）'}")

# ---------- 1. txn ----------
str_cols = ["card_no", "cst_accno", "cntrprt_card_no", "cntrprt_name",
            "cntrprt_clear_type", "txn_channel_type", "txn_cgycd", "smy_cd"]
ov = {c: pl.Utf8 for c in str_cols}
ov.update({"accno_txn_sn": pl.Int64, "acct_bal": pl.Float64, "txn_amt": pl.Float64,
           "inflow_amt": pl.Float64, "outflow_amt": pl.Float64,
           "cntrprt_prvt_ind": pl.Float64, "rmrk_recharge_withdrawal_ind": pl.Float64,
           "rmrk_short_ind": pl.Float64, "rmrk_e_credit_ind": pl.Float64})

txn_parts = [pl.read_csv(os.path.join(TRAIN_DIR, "txn_info_train_testa.csv"),
                         schema_overrides=ov)]
if HAS_TESTB:
    p = os.path.join(TESTB_DIR, "txn_info_testb.csv")
    txn_parts.append(pl.read_csv(p, schema_overrides=ov))
    print(f"  + txn_info_testb: {txn_parts[-1].shape}")
txn = pl.concat(txn_parts, how="vertical")
txn = txn.with_columns([
    pl.col("tms").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
    pl.col("stm_dt").str.to_date("%Y-%m-%d", strict=False),
])
txn.write_parquet(P("txn.parquet"))
print(f"txn(合并后): {txn.shape}  {time.time()-t0:.0f}s")

# ---------- 2. 其余表 ----------
def conv_concat(train_path, testb_path, name, ovv=None):
    parts = [pl.read_csv(train_path, schema_overrides=ovv or {})]
    if HAS_TESTB and os.path.exists(testb_path):
        parts.append(pl.read_csv(testb_path, schema_overrides=ovv or {}))
        print(f"  + {os.path.basename(testb_path)}: {parts[-1].shape}")
    df = pl.concat(parts, how="vertical")
    df.write_parquet(P(name + ".parquet"))
    return df

card = conv_concat(
    os.path.join(TRAIN_DIR, "card_info_train_testa.csv"),
    os.path.join(TESTB_DIR, "card_info_testb.csv"), "card",
    {"card_no": pl.Utf8, "prim_cst_accno": pl.Utf8, "cst_id": pl.Utf8,
     "crdisu_lvl1_insid": pl.Utf8, "crdisu_lvl2_insid": pl.Utf8})
cst = conv_concat(
    os.path.join(TRAIN_DIR, "cst_info_train_testa.csv"),
    os.path.join(TESTB_DIR, "cst_info_testb.csv"), "cst",
    {"cst_id": pl.Utf8, "gender": pl.Utf8, "marital_status": pl.Utf8,
     "occupation": pl.Utf8, "industry": pl.Utf8})
accno = conv_concat(
    os.path.join(TRAIN_DIR, "accno_info_train_testa.csv"),
    os.path.join(TESTB_DIR, "accno_info_testb.csv"), "accno",
    {"card_no": pl.Utf8, "cst_accno": pl.Utf8})

# ⚠ train.csv 的 label 是 "0.0"/"1.0" 浮点文本，直接按 Int8 读会得到 null
train = pl.read_csv(os.path.join(TRAIN_DIR, "train.csv"),
                    schema_overrides={"card_no": pl.Utf8, "label": pl.Float64})
train = train.with_columns(pl.col("label").cast(pl.Int8))
train.write_parquet(P("train.parquet"))

testa = pl.read_csv(os.path.join(TESTA_DIR, "testa.csv"),
                    schema_overrides={"card_no": pl.Utf8})
testa.write_parquet(P("testa.parquet"))

testb = None
if HAS_TESTB:
    testb = pl.read_csv(os.path.join(TESTB_DIR, "testb.csv"),
                        schema_overrides={"card_no": pl.Utf8})
    testb.write_parquet(P("testb.parquet"))
    # 卡号不应与 train/testA 重叠，重叠说明数据放置有误
    overlap = testb.join(pl.concat([train.select("card_no"),
                                    testa.select("card_no")]),
                         on="card_no", how="semi").height
    assert overlap == 0, f"testB 卡号与 train/testA 重叠 {overlap} 个，请检查数据"

print(f"train={train.shape} testa={testa.shape}"
      f"{f' testb={testb.shape}' if testb is not None else ''}"
      f" 欺诈率={train['label'].mean():.6f}")

# ---------- 3. 覆盖性检查 ----------
txn_cards = txn.select(pl.col("card_no").unique())
print("txn 唯一卡数:", txn_cards.height)
print("train 无流水卡:", train.join(txn_cards, on="card_no", how="anti").height)
print("testa 无流水卡:", testa.join(txn_cards, on="card_no", how="anti").height)
if testb is not None:
    print("testb 无流水卡:", testb.join(txn_cards, on="card_no", how="anti").height)

# ---------- 4. 时间锚点 ----------
anchor = txn.group_by("card_no").agg([
    pl.col("stm_dt").max().alias("last_dt"),
    pl.col("stm_dt").min().alias("first_dt"),
    pl.len().alias("n_txn")])
anchor = anchor.with_columns(
    (pl.col("last_dt") - pl.col("first_dt")).dt.total_days().alias("span_days"))
mem_parts = [
    train.select("card_no", pl.col("label").cast(pl.Utf8).alias("grp")),
    testa.select("card_no", pl.lit("testa").alias("grp"))]
if testb is not None:
    mem_parts.append(testb.select("card_no", pl.lit("testb").alias("grp")))
mem = pl.concat(mem_parts)
anchor = anchor.join(mem, on="card_no", how="left")
anchor.write_parquet(P("anchor.parquet"))
print("末笔日期范围:", anchor["last_dt"].min(), "~", anchor["last_dt"].max())
if testb is not None:
    print("testb 末笔日期范围:",
          anchor.filter(pl.col("grp") == "testb")["last_dt"].min(), "~",
          anchor.filter(pl.col("grp") == "testb")["last_dt"].max())
print(f"01 完成 {time.time()-t0:.0f}s")
