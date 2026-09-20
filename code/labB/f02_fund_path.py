# -*- coding: utf-8 -*-
"""f02：账户内资金路径（FIFO 金额匹配）+ 全窗口异常片段

【与现有特征的区别（已核实）】
现有 out_after_in_sec / fio_min / n_fastout_1h 只看「出金距最近一笔入金多久」，
不检验**金额是否对得上**，且用 .over("card_no") 跨账户拼接。本模块在
(card_no, cst_accno) 内按金额做 FIFO 匹配，回答的是「这笔钱多久被转走、
转给了谁、转走了多少」，属于金额层面的流向信息，与 f01 的事件类型组合无关。

【FIFO 匹配的向量化】
逐账户 Python 循环太慢。用累积和区间重叠代替：
  入金 i 占据累积区间 [A(i-1), A(i))，出金 j 消耗 [B(j-1), B(j))。
  对每份入金，用 searchsorted 找第一个使 B(j) >= A(i) 的出金 j，
  该 j 的时刻即这份入金被**完全**转走的时刻。
按账户连续排序后，组内累积和 + 组前缀和 = 全局累积和，故可一次性全局
searchsorted，再校验命中的出金是否仍属同一账户（否则记为未覆盖）。
这是"资金路径代理特征"，不是真实逐元资金追踪。

【两个必须处理对的边界】
  · 末段观察不足：窗口末尾的入金天然来不及被转走。算 cov_Xh 时只把
    「剩余观察时长 >= X」的入金计入**分母**，否则系统性低估覆盖率；
    分母为空则该列写 NaN 并由支持量列标记
  · 期初余额：出金可能来自窗口前的余额。不依赖 acct_bal（AB-thin 已暴露
    余额残差问题），改用 F_out_beyond_in_share 显式标记超出入金的出金占比

【片段划分的口径】
片段按**时间聚集**划分（间隔 > SEG_GAP_SEC 即断开），不涉及前后配对，
故在卡内按 tms 划分是合理的，不违反"不跨账户拼接序列"——后者约束的是
bigram 那类相邻配对。片段特征扫全窗口，不只看末日末笔。

用法: PYTHONPATH=code python code/labB/f02_fund_path.py testb
输出: data/interim_labB/f02_fund.parquet（+ manifest）
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P
from lab_common import LP, data_manifest, load_cached, save_cached

t0 = time.time()
TARGET = sys.argv[1] if len(sys.argv) > 1 else "testb"

FEATURE_VERSION = "f02_fund_v1"
HORIZONS = [("5m", 300), ("30m", 1800), ("2h", 7200), ("24h", 86400)]
PARAMS = dict(SEG_GAP_SEC=6 * 3600, HORIZONS=[h[1] for h in HORIZONS])


def fifo_match(df):
    """账户内 FIFO 匹配。df 需已按 (acc_id, tms, accno_txn_sn) 排序。
    返回每份入金的：覆盖时刻、是否被完全覆盖、覆盖它的出金行号。"""
    is_in = df["inflow_amt"].values > 0
    is_out = df["outflow_amt"].values > 0
    acc = df["acc_id"].values

    ins_idx = np.flatnonzero(is_in)
    out_idx = np.flatnonzero(is_out)
    if len(ins_idx) == 0 or len(out_idx) == 0:
        return ins_idx, np.array([], int), np.zeros(len(ins_idx), bool)

    in_amt = df["inflow_amt"].values[ins_idx]
    out_amt = df["outflow_amt"].values[out_idx]
    in_acc, out_acc = acc[ins_idx], acc[out_idx]

    # 组内累积和 = 全局累积和 - 该组之前的全局累积和（按账户连续排序后成立）
    def within_cum(vals, grp):
        c = np.cumsum(vals)
        first = np.r_[True, grp[1:] != grp[:-1]]
        base = np.repeat(np.r_[0.0, c[np.flatnonzero(first)[1:] - 1]],
                         np.diff(np.r_[np.flatnonzero(first), len(grp)]))
        return c, base

    in_cum, in_base = within_cum(in_amt, in_acc)
    out_cum, out_base = within_cum(out_amt, out_acc)

    # 该入金所属账户的出金全局前缀（账户无出金则取相邻账户前缀，随后校验剔除）
    acc_out_base = pd.Series(out_base, index=out_acc).groupby(level=0).first()
    shift = acc_out_base.reindex(in_acc).values
    has_out = ~np.isnan(shift)
    # 累积和相等时，末位浮点误差会把"恰好覆盖"误判为未覆盖，留 1e-6 元容差
    target = (in_cum - in_base) - 1e-6 + np.nan_to_num(shift)

    pos = np.searchsorted(out_cum, target, side="left")
    pos_c = np.clip(pos, 0, len(out_cum) - 1)
    # 命中的出金必须仍属同一账户，且未越过数组末尾
    covered = has_out & (pos < len(out_cum)) & (out_acc[pos_c] == in_acc)
    return ins_idx, out_idx[pos_c], covered


def build(txn_path=None):
    t = pl.read_parquet(txn_path or P("txn.parquet"),
                        columns=["card_no", "cst_accno", "tms", "accno_txn_sn",
                                 "inflow_amt", "outflow_amt", "cntrprt_card_no"])
    t = t.sort(["card_no", "cst_accno", "tms", "accno_txn_sn"])
    # ⚠ polars 的 Datetime 默认微秒精度，手写 //10**9 会把时间戳压缩 1000 倍
    #   且不报错（自检曾因此让全部 delay/obs 变成 0）。用 dt.epoch 显式取秒。
    t = t.with_columns(pl.col("tms").dt.epoch(time_unit="s").alias("ts"))
    df = t.to_pandas()
    df["card_no"] = df["card_no"].astype(str)
    df["acc_id"] = (df["card_no"] + "\x01"
                    + df["cst_accno"].astype(str)).factorize()[0]
    print(f"  载入 {len(df):,} 行 / {df['acc_id'].nunique():,} 账户"
          f"   {time.time()-t0:.0f}s")

    ins_idx, cov_out_idx, covered = fifo_match(df)
    print(f"  FIFO 匹配完成：入金 {len(ins_idx):,} 笔，"
          f"完全覆盖 {int(covered.sum()):,} 笔   {time.time()-t0:.0f}s")

    card = df["card_no"].values
    ts = df["ts"].values
    # 每个账户的末笔时刻 -> 每份入金的剩余观察时长
    acc_last = pd.Series(ts).groupby(df["acc_id"].values).max()
    in_card = card[ins_idx]
    in_ts = ts[ins_idx]
    in_amt = df["inflow_amt"].values[ins_idx]
    obs_sec = acc_last.reindex(df["acc_id"].values[ins_idx]).values - in_ts
    # 累积和匹配不含时序约束：账户先花期初余额再入金时，会把**入金之前**的
    # 出金错配为"覆盖"，产生负延迟。这类匹配一律判为未覆盖，期初余额的影响
    # 另由 F_out_beyond_in_share 单独刻画。
    raw_delay = ts[cov_out_idx] - in_ts
    covered = covered & (raw_delay >= 0)
    delay = np.where(covered, raw_delay, np.nan)

    rec = pd.DataFrame({"card_no": in_card, "amt": in_amt,
                        "obs": obs_sec, "delay": delay})
    out = pd.DataFrame({"card_no": pd.unique(card)}).set_index("card_no")

    # ---- 覆盖率：分母只含观察时长足够的入金 ----
    for tag, h in HORIZONS:
        sub = rec[rec["obs"] >= h]
        den = sub.groupby("card_no")["amt"].sum()
        num = (sub.assign(w=np.where(sub["delay"] <= h, sub["amt"], 0.0))
               .groupby("card_no")["w"].sum())
        out[f"F_cov_{tag}"] = (num / den).replace([np.inf, -np.inf], np.nan)
        out[f"F_nobs_{tag}"] = sub.groupby("card_no").size()
    out[[f"F_nobs_{tag}" for tag, _ in HORIZONS]] = out[
        [f"F_nobs_{tag}" for tag, _ in HORIZONS]].fillna(0)

    # ---- 转走 50%/90% 入金额所需时间（按延迟排序的金额加权分位）----
    r = rec.dropna(subset=["delay"]).sort_values(["card_no", "delay"])
    if len(r):
        g = r.groupby("card_no")["amt"]
        r["cum"] = g.cumsum()
        r["tot"] = g.transform("sum")
        r["frac"] = r["cum"] / r["tot"]
        for q, name in ((0.5, "F_t50_h"), (0.9, "F_t90_h")):
            hit = r[r["frac"] >= q].groupby("card_no")["delay"].first()
            out[name] = hit / 3600.0
        out["F_cov_delay_p50_h"] = (r.groupby("card_no")["delay"].median()
                                    / 3600.0)
    out["F_uncov_share"] = 1.0 - (
        rec.assign(w=np.where(rec["delay"].notna(), rec["amt"], 0.0))
        .groupby("card_no")["w"].sum() / rec.groupby("card_no")["amt"].sum())
    out["F_n_in"] = rec.groupby("card_no").size()

    # ---- 期初余额代理：出金超出入金的部分 ----
    tot = df.groupby("card_no")[["inflow_amt", "outflow_amt"]].sum()
    out["F_out_beyond_in_share"] = (
        (tot["outflow_amt"] - tot["inflow_amt"]).clip(lower=0)
        / tot["outflow_amt"].replace(0, np.nan))

    # ---- 覆盖出金的对手：数量与新对手金额占比 ----
    cp = df["cntrprt_card_no"].astype(str).values
    first_seen = pd.Series(ts).groupby(cp).transform("min").values
    cov_rows = cov_out_idx[covered]
    if len(cov_rows):
        cd = pd.DataFrame({
            "card_no": card[cov_rows], "cp": cp[cov_rows],
            "amt": df["outflow_amt"].values[cov_rows],
            "is_new": (ts[cov_rows] - first_seen[cov_rows]) == 0})
        out["F_cov_dst_n"] = cd.groupby("card_no")["cp"].nunique()
        out["F_cov_new_dst_share"] = (
            cd[cd["is_new"]].groupby("card_no")["amt"].sum()
            / cd.groupby("card_no")["amt"].sum())
    print(f"  资金路径特征完成   {time.time()-t0:.0f}s")

    # ---- 全窗口异常片段（卡内按时间聚集划分）----
    s = df[["card_no", "ts", "acc_id"]].copy()
    s["amt"] = df["inflow_amt"].values + df["outflow_amt"].values
    s["cp"] = cp
    s = s.sort_values(["card_no", "ts"])
    gap = s.groupby("card_no")["ts"].diff()
    s["seg"] = ((gap.isna()) | (gap > PARAMS["SEG_GAP_SEC"])).cumsum()
    seg = s.groupby(["card_no", "seg"]).agg(
        n=("ts", "size"), t0_=("ts", "min"), t1=("ts", "max"),
        amt=("amt", "sum"), ncp=("cp", "nunique"))
    seg["hours"] = (seg["t1"] - seg["t0_"]) / 3600.0
    seg["rate"] = seg["n"] / (seg["hours"] + 1.0 / 60)   # 笔/小时，防除零
    card_tot = seg.groupby(level=0).agg(
        seg_n=("n", "size"), n_all=("n", "sum"), amt_all=("amt", "sum"),
        h_all=("hours", "sum"))
    top = seg.sort_values("rate", ascending=False).groupby(level=0).first()
    card_last = pd.Series(ts).groupby(card).max()
    out["F_seg_n"] = card_tot["seg_n"]
    out["F_seg_max_rate"] = top["rate"]
    out["F_seg_rate_ratio"] = top["rate"] / (
        card_tot["n_all"] / (card_tot["h_all"] + 1.0 / 60))
    out["F_seg_max_amt_share"] = top["amt"] / card_tot["amt_all"]
    out["F_seg_max_cp_per_txn"] = top["ncp"] / top["n"]
    out["F_seg_max_rel_pos"] = (card_last - top["t1"]) / 3600.0  # 距末笔小时数
    out["F_span_h"] = (card_last - pd.Series(ts).groupby(card).min()) / 3600.0
    print(f"  异常片段特征完成   {time.time()-t0:.0f}s")

    out = out.reset_index()
    for c in out.columns:
        if c != "card_no":
            out[c] = out[c].astype(np.float32)
    return out


def main():
    mani = data_manifest(TARGET, FEATURE_VERSION, PARAMS)
    if load_cached("f02_fund.parquet", mani) is not None:
        print("f02 复用缓存")
        return
    df = build()
    fcols = [c for c in df.columns if c.startswith("F_")]
    print(f"\nf02 共 {len(fcols)} 个特征列，{len(df)} 张卡")
    save_cached(df, "f02_fund.parquet", mani)
    print(f"完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
