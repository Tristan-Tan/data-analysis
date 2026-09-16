# -*- coding: utf-8 -*-
"""团伙图平滑：先在 train 上验证机制，再决定要不要用在 testB 上

【动机】
三次提交锁死 662~663/754，距第一名约 678/754 差 15 张。现有特征体系内的微调
已证明全在噪声内，需要新信息源。exp5 用的是「该卡对手关联过多少 train 里的
已知欺诈卡」，而 **testB 这 75400 张卡彼此之间的关联从未被使用**。电诈系团伙
作案、常共享中转账号。

【上一版跑出来的关键事实】
  · 邻居均分 vs 自身分数 Spearman = 0.2658（相关性存在）
  · top754 边界处分数跨度仅占全量极差的 0.027%（边界卡挤得极密）
  · 因此 alpha=0.1 就洗掉 65% 的 top754 —— alpha 网格必须细到 0.005 量级

【本版新增的关键验证：Spearman 0.27 未必等于有增量信息】
邻居分数与自身分数相关，可能是两种完全不同的原因：
  A 团伙内的卡真的都是欺诈（标签层面聚集）-> 邻居分数是新信息，平滑有用
  B 团伙内的卡特征相似、模型自然给了相似分数（模型已捕获）-> 无增量，
    平滑只是重复模型已知的东西，还会抹平个体差异
0.2658 这个数字本身区分不了 A 和 B，**必须在 train 上用真标签验证**：
用同样的方法平滑 train 的 OOF 分数，看 top1_f1 是升还是降。升=A，降=B。

前置: 已跑过 10_train_models.py testb 与 no_silence_variant.py
用法: PYTHONPATH=code python code/explore/graph_smooth.py
      PYTHONPATH=code python code/explore/graph_smooth.py 20   # 自定义度数上限
耗时: 约 5~10 分钟（train 侧图比 testB 侧大，是主要开销）
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P, PRED_DIR, TOP_FRAC, top1_f1

t0 = time.time()
DEG_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 50
ALPHAS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.2]
FAMILIES = ["dart", "abthin_lgbn", "cat"]


def R(v):
    return rankdata(v) / len(v)


def load_members(prefix):
    """载入五族成员（三族 + nosil partial/full），返回列表"""
    out = []
    for m in FAMILIES:
        out.append(np.load(P(f"{prefix}_{m}.npy")))
    for mode in ("partial", "full"):
        for f in [P(f"{prefix}_nosil_{mode}.npy"),
                  P(f"{prefix}_nosil.npy") if mode == "full" else None]:
            if f and os.path.exists(f):
                out.append(np.load(f))
                break
    return out


def neighbor_mean(card_ids, base_scores, deg_max):
    """共享低度数对手即为邻居。高度数对手（支付宝/美团这类公共账号）必须
    过滤，否则图一稠密信息就被抹平。返回 (邻居均分, 是否有邻居)。"""
    card_set = set(card_ids)
    t = pl.read_parquet(P("txn.parquet"),
                        columns=["card_no", "cntrprt_card_no", "cntrprt_name"])
    t = t.filter(pl.col("card_no").is_in(card_set))
    idx = {c: i for i, c in enumerate(card_ids)}
    nsum = np.zeros(len(card_ids))
    ncnt = np.zeros(len(card_ids))
    for keycol in ("cntrprt_card_no", "cntrprt_name"):
        e = (t.filter(pl.col(keycol).is_not_null())
             .select(["card_no", keycol]).unique().rename({keycol: "key"}))
        deg = e.group_by("key").agg(pl.len().alias("deg"))
        keep = deg.filter((pl.col("deg") >= 2) & (pl.col("deg") <= deg_max))
        e = e.join(keep.select("key"), on="key", how="semi").to_pandas()
        print(f"    {keycol}: 有效对手 {keep.height} 个，边 {len(e)} 条"
              f"   {time.time()-t0:.0f}s")
        e["pos"] = e["card_no"].map(idx)
        e["score"] = base_scores[e["pos"].values]
        g = e.groupby("key")["score"].agg(["sum", "count"])
        e = e.join(g, on="key")
        np.add.at(nsum, e["pos"].values, (e["sum"] - e["score"]).values)
        np.add.at(ncnt, e["pos"].values, (e["count"] - 1).values)
    has = ncnt > 0
    return np.where(has, nsum / np.maximum(ncnt, 1), base_scores), has


# ==================== 段一：train 侧验证（决定这条路走不走） ====================
print("=" * 72)
print("【段一】train 侧验证：图平滑到底有没有增量信息")

tr = pd.read_parquet(P("train.parquet"))
tr["card_no"] = tr["card_no"].astype(str)
y = tr["label"].astype(float).astype(np.int8).values
oof_base = R(np.mean([R(v) for v in load_members("oof")], axis=0))
_, tp_base = top1_f1(y, oof_base)
print(f"  基线 OOF {tp_base}/3000（五族融合）   {time.time()-t0:.0f}s")

nbr_tr, has_tr = neighbor_mean(tr["card_no"].values, oof_base, DEG_MAX)
print(f"  有邻居的卡 {int(has_tr.sum())}/{len(tr)}（{has_tr.mean():.2%}）")
rho_tr = spearmanr(oof_base[has_tr], nbr_tr[has_tr]).correlation
print(f"  邻居均分 vs 自身分数 Spearman = {rho_tr:.4f}")

# 只在有邻居的卡里看真实欺诈率与邻居分数的关系（A/B 之争的直接证据）
hi = has_tr & (nbr_tr >= np.quantile(nbr_tr[has_tr], 0.99))
print(f"  邻居均分 top1% 的卡（{int(hi.sum())} 张）真实欺诈率: "
      f"{y[hi].mean():.4%}  vs 全局 {y.mean():.4%}")

rows = []
for a in ALPHAS:
    s = R((1 - a) * R(oof_base) + a * R(nbr_tr))
    _, tp = top1_f1(y, s)
    rows.append({"alpha": a, "OOF": f"{tp}/3000", "较基线": tp - tp_base})
print("\n  -- train 侧平滑效果（真标签评估）--")
print(pd.DataFrame(rows).to_string(index=False))

best = max(rows, key=lambda r: r["较基线"])
print(f"\n  最佳 alpha={best['alpha']}，较基线 {best['较基线']:+d} 张")
if best["较基线"] <= 0:
    print("  => 平滑在真标签上没有收益：邻居分数属于「模型已捕获」而非新信息，"
          "\n     这条路应当放弃，不要浪费提交次数。下面 testB 侧仅供参考。")
else:
    print(f"  => 平滑确有增量信息，建议用 alpha={best['alpha']} 附近的值做 testB")

# ==================== 段二：testB 侧应用 ====================
print("\n" + "=" * 72)
print("【段二】testB 侧应用")

te = pd.read_parquet(P("testb.parquet"))
te["card_no"] = te["card_no"].astype(str)
base = R(np.mean([R(v) for v in load_members("te")], axis=0))

sil = pd.read_parquet(P("silence_feat.parquet"), columns=["card_no", "sil_days"])
sil["card_no"] = sil["card_no"].astype(str)
hard = te.merge(sil, on="card_no", how="left")["sil_days"].values >= 32
k = int(len(te) * TOP_FRAC)

srt = np.sort(base)[::-1]
print(f"  边界密集度：第 {k-10}~{k+10} 名分数跨度占全量极差 "
      f"{(srt[max(k-11,0)] - srt[min(k+9,len(srt)-1)]) / (srt[0]-srt[-1]):.3%}")

nbr_te, has_te = neighbor_mean(te["card_no"].values, base, DEG_MAX)
print(f"  有邻居的卡 {int(has_te.sum())}/{len(te)}（{has_te.mean():.2%}）")
print(f"  邻居均分 vs 自身分数 Spearman = "
      f"{spearmanr(base[has_te], nbr_te[has_te]).correlation:.4f}")

base_top = set(np.argsort(-base)[:k])
rows = []
for a in ALPHAS:
    s = R((1 - a) * R(base) + a * R(nbr_te))
    if hard.any():
        s[hard] = 1.0
    swap = k - len(set(np.argsort(-s)[:k]) & base_top)
    rows.append({"alpha": a, f"top{k}换动": swap})
    sub = te[["card_no"]].copy()
    sub["score"] = s
    sub.to_csv(os.path.join(
        PRED_DIR, f"prediction_resultB_gs_deg{DEG_MAX}_a{a}.csv"), index=False)
print("\n  -- testB 侧换动张数 --")
print(pd.DataFrame(rows).to_string(index=False))

print(f"""
【怎么用】
  1. 先看段一：train 侧平滑若拿不到正收益，直接放弃，别提交
  2. 若有正收益，取段一最佳 alpha，在段二找对应的候选文件提交
     （文件名 prediction_resultB_gs_deg{DEG_MAX}_a<alpha>.csv）
  3. 段一最佳 alpha 对应的 testB 换动张数若超过 200，说明两侧图密度差异大，
     可下调 alpha 到换动 50~150 的档位，稳一点
""")
print(f"全部完成 {time.time()-t0:.0f}s")
