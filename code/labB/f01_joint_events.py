# -*- coding: utf-8 -*-
"""f01：交易联合事件与账户内序列特征（B 榜实验第一优先级）

【为什么是新信息】（已核实，非沿用结论）
现有 405 维里，摘要/渠道/方向全部是**各自的边际占比**：
    ch{code}_share（02）、smy_{code}_share（03）、smy2_/ch2_（05）
grep 确认**没有任何联合统计**。序列侧只有一个 io_flip（04），且用的是
`.over("card_no")` —— 与 02 的 forward_fill/shift 一样**跨账户拼接**。
本模块的联合事件与账户内 bigram 因此是全新信息，并顺带修正跨账户缺陷。

【特征组】（约 98 列，不堆数百列）
  A 方向×摘要  全窗口 J_ds_*      / 末 N 笔 J_dst_*
  B 方向×渠道  全窗口 J_dc_*      / 末 N 笔 J_dct_*
  C 账户内相邻事件 bigram：方向对 × 时间间隔档 J_bg_*
  D 入→出 金额比例档 J_ioratio_*
  E 支持量：J_n_bigram / J_n_io / J_tail_n

【口径与边界处理】
  · 词表仅用**无监督覆盖度**（覆盖卡数 >= MIN_CARDS 后取 TopN）筛选，
    不使用 label，因此不存在验证泄漏
  · bigram 严格在 (card_no, cst_accno) 内按 (tms, accno_txn_sn) 排序取
    相邻笔，**不跨账户**；间隔超过 GAP_BREAK_SEC 视为片段断开，不成对
  · 分母不足的卡（无有效 bigram / 无 I→O 对）对应占比列写 **NaN** 而非 0，
    由 LightGBM 原生处理；并用支持量列显式标记，避免制造虚假极值
  · 末 N 笔口径沿用现有 04/05 的 rn_end（按 card_no 倒序），保证可对比

用法: PYTHONPATH=code python code/labB/f01_joint_events.py testb
输出: data/interim_labB/f01_joint.parquet（+ manifest，不碰正式 interim）
"""
import os
import sys
import time

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P
from lab_common import LP, data_manifest, load_cached, save_cached

t0 = time.time()
TARGET = sys.argv[1] if len(sys.argv) > 1 else "testb"

FEATURE_VERSION = "f01_joint_v1"
PARAMS = dict(TOP_DIR_SMY=20, TOP_DIR_CH=15, MIN_CARDS=300,
              TAIL_N=10, GAP_BREAK_SEC=3 * 86400)

GAP_BINS = ["5m", "30m", "2h", "1d", "gt1d"]
BG_KEYS = [f"{a}{b}_{g}" for a in "IO" for b in "IO" for g in GAP_BINS]
RT_KEYS = ["lt50", "50_95", "95_105", "105_200", "gt200"]


def _dir_expr():
    return (pl.when(pl.col("inflow_amt") > 0).then(pl.lit("I"))
            .when(pl.col("outflow_amt") > 0).then(pl.lit("O"))
            .otherwise(pl.lit("Z")))


def _gap_bin_expr(c="gap"):
    return (pl.when(pl.col(c) < 300).then(pl.lit("5m"))
            .when(pl.col(c) < 1800).then(pl.lit("30m"))
            .when(pl.col(c) < 7200).then(pl.lit("2h"))
            .when(pl.col(c) < 86400).then(pl.lit("1d"))
            .otherwise(pl.lit("gt1d")))


def _ratio_bin_expr(c="ratio"):
    return (pl.when(pl.col(c) < 0.5).then(pl.lit("lt50"))
            .when(pl.col(c) < 0.95).then(pl.lit("50_95"))
            .when(pl.col(c) < 1.05).then(pl.lit("95_105"))
            .when(pl.col(c) < 2.0).then(pl.lit("105_200"))
            .otherwise(pl.lit("gt200")))


def pick_vocab(t, col, top_n, min_cards):
    """无监督词表：覆盖卡数 >= min_cards，按覆盖卡数取 TopN"""
    cov = (t.group_by(col).agg(pl.col("card_no").n_unique().alias("nc"))
           .filter((pl.col("nc") >= min_cards) & pl.col(col).is_not_null())
           .sort("nc", descending=True))
    vocab = cov[col].to_list()[:top_n]
    print(f"    {col}: 满足覆盖>={min_cards} 的组合 {cov.height} 个，取 Top{len(vocab)}")
    return vocab


def build(txn_path=None):
    """构造特征。txn_path 可指向合成数据，供 selftest 调用。"""
    t = pl.read_parquet(txn_path or P("txn.parquet"),
                        columns=["card_no", "cst_accno", "tms", "accno_txn_sn",
                                 "inflow_amt", "outflow_amt", "txn_amt",
                                 "smy_cd", "txn_channel_type"])
    t = t.with_columns(_dir_expr().alias("dir"))
    t = t.with_columns([
        (pl.col("dir") + pl.lit("_") + pl.col("smy_cd").fill_null("NA")).alias("ds"),
        (pl.col("dir") + pl.lit("_")
         + pl.col("txn_channel_type").fill_null("NA")).alias("dc"),
    ])

    print("  [词表]（无监督覆盖度筛选，不使用 label）")
    ds_vocab = pick_vocab(t, "ds", PARAMS["TOP_DIR_SMY"], PARAMS["MIN_CARDS"])
    dc_vocab = pick_vocab(t, "dc", PARAMS["TOP_DIR_CH"], PARAMS["MIN_CARDS"])

    # ---- A/B 全窗口 ----
    t = t.sort(["card_no", "tms", "accno_txn_sn"])
    out = t.group_by("card_no").agg(
        [(pl.col("ds") == v).mean().alias(f"J_ds_{v}") for v in ds_vocab]
        + [(pl.col("dc") == v).mean().alias(f"J_dc_{v}") for v in dc_vocab]
        + [pl.len().alias("J_n_txn_chk")])
    print(f"  全窗口联合占比 {len(ds_vocab) + len(dc_vocab)} 列"
          f"   {time.time()-t0:.0f}s")

    # ---- A/B 末 N 笔（rn_end 口径与现有 04/05 一致）----
    t = t.with_columns(
        (pl.len().over("card_no") - 1
         - pl.int_range(pl.len()).over("card_no")).alias("rn_end"))
    tail = t.filter(pl.col("rn_end") < PARAMS["TAIL_N"])
    gt = tail.group_by("card_no").agg(
        [(pl.col("ds") == v).mean().alias(f"J_dst_{v}") for v in ds_vocab]
        + [(pl.col("dc") == v).mean().alias(f"J_dct_{v}") for v in dc_vocab]
        + [pl.len().alias("J_tail_n")])
    out = out.join(gt, on="card_no", how="left")
    print(f"  末 {PARAMS['TAIL_N']} 笔联合占比   {time.time()-t0:.0f}s")

    # ---- C 账户内 bigram（不跨账户；长间隔断开）----
    tb = t.sort(["card_no", "cst_accno", "tms", "accno_txn_sn"])
    acc = ["card_no", "cst_accno"]
    tb = tb.with_columns([
        pl.col("dir").shift(1).over(acc).alias("pdir"),
        pl.col("tms").shift(1).over(acc).alias("ptms"),
        pl.col("txn_amt").shift(1).over(acc).alias("pamt"),
    ])
    tb = tb.with_columns(
        (pl.col("tms") - pl.col("ptms")).dt.total_seconds().alias("gap"))
    tb = tb.filter(pl.col("pdir").is_not_null()
                   & (pl.col("gap") <= PARAMS["GAP_BREAK_SEC"]))
    tb = tb.with_columns(
        (pl.col("pdir") + pl.col("dir") + pl.lit("_")
         + _gap_bin_expr()).alias("bg"))
    gbg = tb.group_by("card_no").agg(
        [(pl.col("bg") == k).mean().alias(f"J_bg_{k}") for k in BG_KEYS]
        + [pl.len().alias("J_n_bigram")])
    out = out.join(gbg, on="card_no", how="left")
    print(f"  账户内 bigram {len(BG_KEYS)} 列   {time.time()-t0:.0f}s")

    # ---- D 入→出 金额比例档 ----
    io = tb.filter((pl.col("pdir") == "I") & (pl.col("dir") == "O")
                   & (pl.col("pamt") > 0))
    io = io.with_columns((pl.col("txn_amt") / pl.col("pamt")).alias("ratio"))
    io = io.with_columns(_ratio_bin_expr().alias("rbin"))
    gio = io.group_by("card_no").agg(
        [(pl.col("rbin") == k).mean().alias(f"J_ioratio_{k}") for k in RT_KEYS]
        + [pl.len().alias("J_n_io")])
    out = out.join(gio, on="card_no", how="left")
    print(f"  入→出金额比例档 {len(RT_KEYS)} 列   {time.time()-t0:.0f}s")

    # 支持量缺失 -> 0（"没有这种事件"是确定信息）；占比列保持 NaN
    out = out.with_columns([
        pl.col("J_n_bigram").fill_null(0), pl.col("J_n_io").fill_null(0),
        pl.col("J_tail_n").fill_null(0)]).drop("J_n_txn_chk")
    return out


def main():
    mani = data_manifest(TARGET, FEATURE_VERSION, PARAMS)
    cached = load_cached("f01_joint.parquet", mani)
    if cached is not None:
        print(f"f01 复用缓存，{cached.shape}")
        return
    df = build().to_pandas()
    df["card_no"] = df["card_no"].astype(str)
    fcols = [c for c in df.columns if c.startswith("J_")]
    print(f"\nf01 共 {len(fcols)} 个特征列，{len(df)} 张卡")
    save_cached(df, "f01_joint.parquet", mani)
    print(f"完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
