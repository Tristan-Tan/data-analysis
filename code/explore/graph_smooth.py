# -*- coding: utf-8 -*-
"""testB 内部团伙图平滑 —— 引入唯一一块从未使用过的信息

【动机】
三次提交锁死在 662~663/754，距第一名约 678/754 差 15 张。现有特征体系内的
微调（融合结构、权重、屏蔽特征）已被证明全在噪声内，需要新信息源。

exp5 暴露编码用的是「该卡的对手关联过多少 **train 里的已知欺诈卡**」——
利用的是 train 的标签。而 **testB 这 75400 张卡彼此之间的关联从未被使用**。

电诈是团伙作案，一批卡往往同时被控制、共享中转账号。若 testB 内欺诈卡成团
出现，则一张原本排 760 名的卡，如果其关联卡普遍分数很高，它大概率也是欺诈。
边界卡恰恰决定 top754 的命中数。

【做法】图平滑：final = (1-alpha) * 自己的分数 + alpha * 邻居平均分数
关键是必须过滤高度数对手，否则所有卡都连到支付宝/美团这类公共对手上，图
一稠密信息就被抹平。沿用 exp5 低度数变体的成熟做法（deg 上限可调）。

【顺带诊断】top754 边界附近的分数密集程度：边界处卡越密集，说明微小扰动
就能换进换出越多卡，图平滑的收益空间越大。

前置: 已跑过 10_train_models.py testb（需要 te_*.npy）与 predict.py testb
用法: PYTHONPATH=code python code/explore/graph_smooth.py
      PYTHONPATH=code python code/explore/graph_smooth.py 100   # 自定义度数上限
耗时: 约 3~5 分钟
输出: 各 alpha 下的候选提交文件 prediction_result/prediction_resultB_gs*.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P, PRED_DIR, TOP_FRAC

t0 = time.time()
DEG_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 50   # 对手度数上限
ALPHAS = [0.1, 0.2, 0.3, 0.4]
FAMILIES = ["dart", "abthin_lgbn", "cat"]


def R(v):
    return rankdata(v) / len(v)


# ---------------- 基线分数：与已提交的五族版一致 ----------------
te = pd.read_parquet(P("testb.parquet"))
te["card_no"] = te["card_no"].astype(str)
parts = []
for m in FAMILIES:
    parts.append(np.load(P(f"te_{m}.npy")))
for mode in ("partial", "full"):
    for f in (P(f"te_nosil_{mode}.npy"), P("te_nosil.npy") if mode == "full" else None):
        if f and os.path.exists(f):
            parts.append(np.load(f))
            break
print(f"融合成员数 {len(parts)}（五族版基线）")
base = R(np.mean([R(v) for v in parts], axis=0))

sil = pd.read_parquet(P("silence_feat.parquet"), columns=["card_no", "sil_days"])
sil["card_no"] = sil["card_no"].astype(str)
sv = te.merge(sil, on="card_no", how="left")["sil_days"].values
hard = sv >= 32

k = int(len(te) * TOP_FRAC)
print(f"testB {len(te)} 张，top{k}   {time.time()-t0:.0f}s")

# ---------------- 诊断：边界附近的分数密集程度 ----------------
srt = np.sort(base)[::-1]
print("\n【边界密集度诊断】")
print(f"  第 {k} 名分数: {srt[k-1]:.6f}")
for w in (10, 30, 50, 100):
    lo, hi = srt[min(k - 1 + w, len(srt) - 1)], srt[max(k - 1 - w, 0)]
    print(f"  第 {k-w}~{k+w} 名分数跨度: {hi - lo:.6f}"
          f"（占全量分数极差的 {(hi - lo) / (srt[0] - srt[-1]):.3%}）")
print("  跨度越小 -> 边界卡挤得越密 -> 轻微扰动就能换动大量卡")

# ---------------- 构建 testB 卡-卡关联图 ----------------
txn = pl.read_parquet(P("txn.parquet"),
                      columns=["card_no", "cntrprt_card_no", "cntrprt_name"])
tb = set(te["card_no"])
txn = txn.filter(pl.col("card_no").is_in(tb))

idx = {c: i for i, c in enumerate(te["card_no"].values)}
nbr_sum = np.zeros(len(te))
nbr_cnt = np.zeros(len(te))

for keycol in ("cntrprt_card_no", "cntrprt_name"):
    e = (txn.filter(pl.col(keycol).is_not_null())
         .select(["card_no", keycol]).unique()
         .rename({keycol: "key"}))
    deg = e.group_by("key").agg(pl.len().alias("deg"))
    # 只保留 2~DEG_MAX：度数 1 无法连边，度数过大是公共对手（支付宝/美团等）
    keep = deg.filter((pl.col("deg") >= 2) & (pl.col("deg") <= DEG_MAX))
    e = e.join(keep.select("key"), on="key", how="semi").to_pandas()
    print(f"\n  {keycol}: 有效对手 {keep.height} 个，关联边 {len(e)} 条"
          f"   {time.time()-t0:.0f}s")

    # 对每个 key，把组内其他卡的分数汇总给本卡（不含自己）
    e["pos"] = e["card_no"].map(idx)
    e["score"] = base[e["pos"].values]
    g = e.groupby("key")["score"].agg(["sum", "count"])
    e = e.join(g, on="key")
    np.add.at(nbr_sum, e["pos"].values, (e["sum"] - e["score"]).values)
    np.add.at(nbr_cnt, e["pos"].values, (e["count"] - 1).values)

has_nbr = nbr_cnt > 0
print(f"\n有邻居的卡: {int(has_nbr.sum())} / {len(te)}"
      f"（{has_nbr.mean():.2%}）   {time.time()-t0:.0f}s")
nbr_mean = np.where(has_nbr, nbr_sum / np.maximum(nbr_cnt, 1), base)

# 邻居分数与自身分数的相关性：高 -> 团伙聚集确实存在
from scipy.stats import spearmanr
rho = spearmanr(base[has_nbr], nbr_mean[has_nbr]).correlation
print(f"邻居均分 vs 自身分数 Spearman = {rho:.4f}"
      "（明显为正说明存在团伙聚集，平滑才有意义）")

# ---------------- 各 alpha 下的平滑结果 ----------------
print("\n" + "=" * 70)
print("【图平滑候选】")
base_top = set(np.argsort(-base)[:k])
rows = []
for a in ALPHAS:
    s = R((1 - a) * R(base) + a * R(nbr_mean))
    if hard.any():
        s[hard] = 1.0
    swap = k - len(set(np.argsort(-s)[:k]) & base_top)
    rows.append({"alpha": a, f"top{k}换动": swap})

    sub = te[["card_no"]].copy()
    sub["score"] = s
    out = os.path.join(PRED_DIR,
                       f"prediction_resultB_gs_deg{DEG_MAX}_a{int(a*100)}.csv")
    sub.to_csv(out, index=False)
print(pd.DataFrame(rows).to_string(index=False))
print(f"\n候选文件已写出到 {PRED_DIR}")
print("""
【怎么选】
  · 先看「邻居均分 vs 自身分数」的 Spearman：若接近 0，说明 testB 内部没有
    可利用的团伙聚集，这条路直接放弃，别浪费提交次数
  · 若明显为正（>0.2），挑换动 30~100 张的 alpha 提交
  · 度数上限可调：deg 上限调小(如 20)图更稀疏、更聚焦真团伙；调大(如 100)
    覆盖更多卡但噪声更大。默认 50
""")
print(f"全部完成 {time.time()-t0:.0f}s")
