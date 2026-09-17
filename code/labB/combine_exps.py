# -*- coding: utf-8 -*-
"""三次实验模型的组合评估（零成本，纯 npy 重组，不重训）

【动机】
f01/f02/exp3 三次实验的单模型 OOF 全为负（-5/-13/-7），但融合层面都为正
（+1/+5/+4）。这是集成多样性的典型表现，值得把它们一起评估一次。

【必须先避开的陷阱】
exp1/exp2/exp3 用的是同一套 lgbn 超参、同一折划分，只是特征集不同，彼此
可能高度相关。若直接 base3 + exp1 + exp2 + exp3 六族等权，lgbn 类模型会占
4/6 权重，**破坏原有族平衡**。因此：
  · 先打印三个 exp 两两的 Spearman 与 top1% 重叠度
  · 组合方案里同时给出「三个 exp 先族内平均成一族」与「各自独立成族」两种，
    由数据决定哪种合理，而不是默认堆权重

【判定】换卡数量不是目标。看的是 OOF 换入正例/换出正例的**比值**与净收益：
若换入正例数不显著高于换出正例数，则属噪声，不应消耗提交次数。

前置: run_exp1/2/3 均已跑过（labB 目录下有 te_exp*.npy 与 oof_exp*.npy）
用法: PYTHONPATH=code python code/labB/combine_exps.py testb
耗时: 约 10~30 秒（只读 npy/parquet，不训练）
"""
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P, FOLD_SEED, N_FOLDS, TOP_FRAC, top1_f1
from lab_common import LP, LAB_PRED, R

t0 = time.time()
TARGET = sys.argv[1] if len(sys.argv) > 1 else "testb"
BASE3 = ["dart", "abthin_lgbn", "cat"]
EXPS = ["exp1", "exp2", "exp3"]
print(f"目标集 = {TARGET}   （若这里不是 testb，请中止并带参数重跑）")

# ---------------- 载入 + 来源核对（要求⑦） ----------------
print("\n【来源核对】所有输入文件的路径与落盘时间")


def _load(path):
    st = os.stat(path)
    print(f"    {os.path.relpath(path, os.getcwd()):<52} "
          f"{st.st_size / 1e6:7.2f}MB  "
          f"{datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M}")
    return np.load(path)


oof, tev = {}, {}
for m in BASE3:
    oof[m] = _load(P(f"oof_{m}.npy"))
    tev[m] = _load(P(f"te_{m}.npy"))
avail = []
for e in EXPS:
    po, pt = LP(f"oof_{e}_{TARGET}.npy"), LP(f"te_{e}_{TARGET}.npy")
    if os.path.exists(po) and os.path.exists(pt):
        oof[e] = _load(po)
        tev[e] = _load(pt)
        avail.append(e)
    else:
        print(f"    缺少 {e} 的 npy（{os.path.basename(po)}），跳过该成员")
print(f"  可用成员: {BASE3 + avail}")

tr = pd.read_parquet(P("train.parquet"))
y = tr["label"].astype(float).astype(np.int8).values
te = pd.read_parquet(P(f"{TARGET}.parquet"))
te["card_no"] = te["card_no"].astype(str)
k_te = int(len(te) * TOP_FRAC)
k_tr = int(len(y) * TOP_FRAC)
print(f"  train {len(y):,} 张（正例 {int(y.sum()):,}，top1%={k_tr}）  "
      f"{TARGET} {len(te):,} 张（top1%={k_te}）")
for m in BASE3:
    assert len(oof[m]) == len(y) and len(tev[m]) == len(te), f"{m} 长度不符"
for e in avail:
    assert len(oof[e]) == len(y) and len(tev[e]) == len(te), f"{e} 长度不符"
print("  长度校验通过（所有成员与 train/{} 行数一致）".format(TARGET))

# ---------------- 相关性：三个 exp 是否过于同质 ----------------
print("\n" + "=" * 74)
print("【相关性】三个 exp 之间（同超参同折，只差特征集，可能高度相关）")


def topk_overlap(a, b, k):
    return len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / k


for i, a in enumerate(avail):
    for b in avail[i + 1:]:
        print(f"  {a} vs {b}:  OOF Spearman="
              f"{spearmanr(oof[a], oof[b]).correlation:.4f}"
              f"   {TARGET} top{k_te} 重叠={topk_overlap(tev[a], tev[b], k_te):.3f}")
print("  参考：三族之间的 OOF Spearman（真正多样的族应明显更低）")
for i, a in enumerate(BASE3):
    for b in BASE3[i + 1:]:
        print(f"  {a} vs {b}:  {spearmanr(oof[a], oof[b]).correlation:.4f}")

# ---------------- 组合方案 ----------------
def blend(members):
    """members: [[成员名...], ...] 每个子列表先族内平均，再按固定族权重等权"""
    fam_o, fam_t = [], []
    for grp in members:
        fam_o.append(np.mean([R(oof[m]) for m in grp], axis=0))
        fam_t.append(np.mean([R(tev[m]) for m in grp], axis=0))
    return (R(np.mean([R(v) for v in fam_o], axis=0)),
            R(np.mean([R(v) for v in fam_t], axis=0)))


BASE_NAME = "base3（已提交 663）"
CAND = {BASE_NAME: [[m] for m in BASE3]}
for e in avail:
    CAND[f"base3 + {e}（{e} 单独成第4族）"] = [[m] for m in BASE3] + [[e]]
if len(avail) >= 2:
    CAND["base3 + exp族（3个exp先族内平均，推荐口径）"] = \
        [[m] for m in BASE3] + [avail]
    CAND["base3 + 各exp独立成族（会改变族权重）"] = \
        [[m] for m in BASE3] + [[e] for e in avail]

# 硬规则：sil_days >= 32 置顶，与 predict.py 完全一致
sil = pd.read_parquet(P("silence_feat.parquet"), columns=["card_no", "sil_days"])
sil["card_no"] = sil["card_no"].astype(str)
sv = te.merge(sil, on="card_no", how="left")["sil_days"].values
hard = sv >= 32
print(f"\n  硬规则 sil_days>=32 命中 {int(hard.sum())} 张"
      f"（缺失 {int(pd.isna(sv).sum())} 张，与 predict.py 口径一致）")

folds = list(StratifiedKFold(N_FOLDS, shuffle=True,
                             random_state=FOLD_SEED).split(np.zeros(len(y)), y))


def per_fold_hits(score):
    out = []
    for _, vi in folds:
        kk = int(len(vi) * TOP_FRAC)
        out.append(int(y[vi][np.argsort(-score[vi])[:kk]].sum()))
    return out


def apply_hard(t):
    if not hard.any():
        return t
    t = t.copy()
    t[hard] = 1.0
    return t


base_o, base_t = blend(CAND[BASE_NAME])
base_t = apply_hard(base_t)
_, tp_base = top1_f1(y, base_o)
fh_base = per_fold_hits(base_o)
base_top = set(np.argsort(-base_t)[:k_te].tolist())
base_rank = rankdata(-base_t, method="ordinal")

print("\n" + "=" * 74)
print(f"【1】OOF 基准 base3 = {tp_base}/{k_tr}   各折 {fh_base}  "
      f"合计 {sum(fh_base)}")

rows = []
detail = {}
for name, members in CAND.items():
    o, t = blend(members)
    t = apply_hard(t)
    _, tp = top1_f1(y, o)
    fh = per_fold_hits(o)
    # 【2】【3】OOF 每侧换卡量与换入/换出正例
    b = np.argsort(-base_o)[:k_tr]
    n = np.argsort(-o)[:k_tr]
    sin, sout = sorted(set(n.tolist()) - set(b.tolist())), \
        sorted(set(b.tolist()) - set(n.tolist()))
    ip = int(y[sin].sum()) if sin else 0
    op = int(y[sout].sum()) if sout else 0
    # 【4】testB 侧换卡量与换入卡的基线原排名
    new_top = np.argsort(-t)[:k_te]
    te_in = sorted(set(new_top.tolist()) - base_top)
    swap_te = len(te_in)
    rk = base_rank[te_in] if te_in else np.array([0])
    rows.append({"组合": name, "OOF": tp, "OOF差": tp - tp_base,
                 "逐折差": str([a - b2 for a, b2 in zip(fh, fh_base)]),
                 "每侧m": len(sin), "换入正例": ip, "换出正例": op,
                 "净": ip - op,
                 f"{TARGET}换动": swap_te,
                 "换入原排名p50": int(np.median(rk)),
                 "换入原排名max": int(rk.max())})
    detail[name] = (o, t)

d = pd.DataFrame(rows)
pd.set_option("display.width", 240)
pd.set_option("display.max_colwidth", 42)
print("\n【2/3/4】各组合对比（OOF 换入/换出正例 + testB 换卡）")
print(d.to_string(index=False))

# ---------------- 【5】边界条件区分能力 ----------------
print("\n【5】边界条件区分能力（基线排名落在该区间的子集内 AUC）")
from sklearn.metrics import roc_auc_score  # noqa: E402
br_oof = rankdata(-base_o, method="ordinal")
for lo, hi in [(1500, 6000), (2500, 6000)]:
    m = (br_oof >= lo) & (br_oof < hi)
    ys = y[m]
    if ys.sum() in (0, len(ys)):
        print(f"  [{lo},{hi}) 正例数 {int(ys.sum())}，跳过")
        continue
    ab = roc_auc_score(ys, base_o[m])
    line = [f"  [{lo},{hi}) n={int(m.sum())} 正例={int(ys.sum())} "
            f"基线AUC={ab:.4f}"]
    for name in CAND:
        if name == BASE_NAME:
            continue
        an = roc_auc_score(ys, detail[name][0][m])
        line.append(f"    {name:<36} {an:.4f}（{an - ab:+.4f}）")
    print("\n".join(line))

# ---------------- 【6】新入选卡由什么支持 ----------------
print("\n【6】新入选卡由哪些成员推上来（该卡在各成员内的 rank 百分位，"
      "1.0=最高分）")
for name in CAND:
    if name == BASE_NAME:
        continue
    _, t = detail[name]
    te_in = sorted(set(np.argsort(-t)[:k_te].tolist()) - base_top)
    if not te_in:
        print(f"  {name}: 无新入选卡")
        continue
    seg = [f"  {name}（{len(te_in)} 张新入选）:"]
    for m in BASE3 + avail:
        pr = R(tev[m])
        seg.append(f"    {m:<14} 新入选卡 p50={np.median(pr[te_in]):.4f} "
                   f"vs 全体 p50=0.5000")
    print("\n".join(seg))

print("""
【判读】
  · 「换入正例 / 换出正例」的比值才是信号强弱的指标。若换入正例 <= 换出正例，
    或二者接近（差 1~3 张在 3000 张规模上属噪声），则该组合只是噪声重排，
    不应消耗提交次数。
  · 若「exp族先族内平均」明显优于「各exp独立成族」，说明三个 exp 确实同质，
    独立成族只是变相给 lgbn 类加权，不是真正的多样性。
  · 【6】中若新入选卡在三个 exp 内的百分位明显高于在 base3 内的百分位，
    说明是 exp 的新增信息把它们推上来的；若两边接近，则只是排名抖动。
  · OOF 与线上在本赛题已被证明脱钩，OOF 正收益不足以单独支撑提交决策；
    需要 OOF 净收益与【5】【6】同时给出一致结论。
""")

# 写出最优候选（按 OOF 净收益）供人工决定，不自动提交
best = d[d["组合"] != BASE_NAME].sort_values(
    ["净", "OOF"], ascending=False)
if len(best):
    name = best.iloc[0]["组合"]
    t = detail[name][1]
    sub = te[["card_no"]].copy()
    sub["score"] = t
    assert (sub["card_no"].values == te["card_no"].values).all()
    out = os.path.join(LAB_PRED, f"pred_{TARGET}_combine_best.csv")
    sub.to_csv(out, index=False)
    net = int(best.iloc[0]["净"])
    print(f"按 OOF 净收益最优的组合：{name}（净 {net:+d}）")
    print(f"候选文件（仅落盘，未提交）: {out}")
    if net <= 0:
        print("  ⚠ 最优组合的 OOF 净收益 <= 0：落盘只为留痕，"
              "**不建议消耗提交次数**")
    elif net <= 3:
        print("  ⚠ 净收益仅 %+d 张（3000 张规模），量级接近噪声；"
              "请结合【5】【6】判断，不要仅凭此提交" % net)

print(f"\n全部完成 {time.time()-t0:.0f}s")
