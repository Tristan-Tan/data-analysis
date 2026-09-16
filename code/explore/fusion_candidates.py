# -*- coding: utf-8 -*-
"""融合结构枚举：不重训，只重新组合已有的族预测

【为什么】
no_silence_variant.py 两种模式训完后陷入两难：
  full    单族 OOF 2633（太弱），四族融合 -18，但 top754 换动 34 张
  partial 单族 OOF 2755（够强，甚至高于 cat 2744），融合 +3，但换动仅 7 张
换动 <20 张意味着提交后涨跌都无法归因，等于白花一次提交次数。

但这些成员都已训好，te_*.npy 全在手上，**重新组合只要几秒**。与其纠结
"追加第四族"这一种结构，不如把各种融合方式枚举一遍，挑一个换动张数落在
有效区间（20~150）且 OOF 不掉的方案再提交。

权重通过重复成员实现（融合公式 R(mean([R(x) for x in members]))，与
predict.py 完全一致）。硬规则 sil_days>=32 置顶同样保持一致。

前置: 已跑过 10_train_models.py testb 与 no_silence_variant.py 的两种模式
用法: PYTHONPATH=code python code/explore/fusion_candidates.py
耗时: 几秒（纯 npy 重组）
输出: 对比表格；并为每个候选写出 prediction_result/prediction_resultB_fuse_*.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P, PRED_DIR, TOP_FRAC, top1_f1
import assemble

t0 = time.time()


def R(v):
    return rankdata(v) / len(v)


def blend(parts):
    return R(np.mean([R(v) for v in parts], axis=0))


# ---------------- 载入各族的 OOF 与 testB 预测 ----------------
need = {"dart": "dart", "abthin": "abthin_lgbn", "cat": "cat"}
oof, tev = {}, {}
for k, name in need.items():
    po, pt = P(f"oof_{name}.npy"), P(f"te_{name}.npy")
    assert os.path.exists(po) and os.path.exists(pt), f"缺少 {name} 的 npy"
    oof[k], tev[k] = np.load(po), np.load(pt)
for mode in ("full", "partial"):
    # 早期版本的 no_silence_variant.py 存的是不带 mode 后缀的文件名（那次跑的
    # 就是 full 模式），这里一并兼容，免得还要手动重命名
    cands = [(P(f"oof_nosil_{mode}.npy"), P(f"te_nosil_{mode}.npy"))]
    if mode == "full":
        cands.append((P("oof_nosil.npy"), P("te_nosil.npy")))
    for po, pt in cands:
        if os.path.exists(po) and os.path.exists(pt):
            oof[mode], tev[mode] = np.load(po), np.load(pt)
            print(f"  {mode:<8} <- {os.path.basename(po)}")
            break
    else:
        print(f"  {mode:<8} <- 未找到 npy，相关候选将跳过")
print(f"已载入成员: {list(oof)}   {time.time()-t0:.0f}s")

tr = pd.read_parquet(P("train.parquet"))
y = tr["label"].astype(float).astype(np.int8).values

# ---------------- 候选融合结构 ----------------
# partial 强但与现有三族同质（换动少），full 弱但真正异构（换动多）。
# 按 full 权重递增排成一条权衡曲线，找「换动刚够 20~150、OOF 掉得最少」的点。
CANDIDATES = {
    "base3(已提交662)":      ["dart", "abthin", "cat"],
    # --- partial 系：强但同质，靠加权也撬不动排序 ---
    "+partial":              ["dart", "abthin", "cat", "partial"],
    "swap_cat->partial":     ["dart", "abthin", "partial"],
    "partial_x2":            ["dart", "abthin", "cat", "partial", "partial"],
    "partial_x3":            ["dart", "abthin", "cat",
                              "partial", "partial", "partial"],
    # --- full 系：按权重递增，换动随之上升、OOF 随之下降 ---
    "+full":                 ["dart", "abthin", "cat", "full"],
    "full_x2":               ["dart", "abthin", "cat", "full", "full"],
    "full_x3":               ["dart", "abthin", "cat", "full", "full", "full"],
    "swap_cat->full":        ["dart", "abthin", "full"],
    # --- 两个新成员同时进来 ---
    "+both(五族)":           ["dart", "abthin", "cat", "partial", "full"],
    "partial+full_x2":       ["dart", "abthin", "cat",
                              "partial", "full", "full"],
    "partial_x2+full":       ["dart", "abthin", "cat",
                              "partial", "partial", "full"],
}
_skipped = [k for k, v in CANDIDATES.items() if not all(m in oof for m in v)]
CANDIDATES = {k: v for k, v in CANDIDATES.items()
              if all(m in oof for m in v)}
if _skipped:
    print(f"⚠ 因成员缺失跳过的候选: {_skipped}")

# ---------------- 硬规则所需的 sil_days ----------------
te = pd.read_parquet(P("testb.parquet"))
te["card_no"] = te["card_no"].astype(str)
sil = pd.read_parquet(P("silence_feat.parquet"),
                      columns=["card_no", "sil_days"])
sil["card_no"] = sil["card_no"].astype(str)
sv = te.merge(sil, on="card_no", how="left")["sil_days"].values
hard = sv >= 32
print(f"硬规则命中 {int(hard.sum())} 张（sil_days >= 32）")


def testb_score(members):
    s = blend([tev[m] for m in members])
    if hard.any():
        s[hard] = 1.0
    return s


k_te = int(len(te) * TOP_FRAC)
BASE = "base3(已提交662)"
base_top = set(np.argsort(-testb_score(CANDIDATES[BASE]))[:k_te])
_, tp_base = top1_f1(y, blend([oof[m] for m in CANDIDATES[BASE]]))

rows = []
scores = {}
for name, members in CANDIDATES.items():
    _, tp = top1_f1(y, blend([oof[m] for m in members]))
    s = testb_score(members)
    scores[name] = s
    swap = k_te - len(set(np.argsort(-s)[:k_te]) & base_top)
    ok = name != BASE and 20 <= swap <= 150 and tp - tp_base >= -10
    rows.append({"候选": name,
                 "OOF": f"{tp}/3000",
                 "OOF差": tp - tp_base,
                 f"top{k_te}换动": swap,
                 "推荐": "★" if ok else ""})

print("\n" + "=" * 78)
print(f"【融合候选对比】（OOF差 以 base3={tp_base} 为基准；"
      "★ = 换动 20~150 且 OOF差 >= -10）")
d = pd.DataFrame(rows)
print(d.to_string(index=False))
n_ok = int((d["推荐"] == "★").sum())
print(f"\n满足条件的候选: {n_ok} 个"
      + ("" if n_ok else "  -> 成员彼此过于相关，融合已无空间，建议停止投入"))

# ---------------- 写出候选提交文件 ----------------
print()
for name, s in scores.items():
    if name.startswith("base3"):
        continue
    tag = (name.replace("+", "plus_").replace("->", "2")
           .replace("(", "_").replace(")", "").replace("五族", "5fam"))
    sub = te[["card_no"]].copy()
    sub["score"] = s
    assert (sub["card_no"].values == te["card_no"].values).all()
    out = os.path.join(PRED_DIR, f"prediction_resultB_fuse_{tag}.csv")
    sub.to_csv(out, index=False)
    print(f"  已写出 {os.path.basename(out)}")

print("""
【怎么选】
  1. 先排除 OOF差 明显为负（< -10）的候选
  2. 在剩下的里挑「换动张数落在 20~150」的——换动太少提交了也看不出结果
  3. 同时满足两条的，优先选 OOF差 >= 0 的
  4. 若没有任何候选同时满足，说明这批成员彼此太相关，再怎么组合也撬不动
     排序，应停止在融合上投入，把剩余时间转向审查材料
""")
print(f"全部完成 {time.time()-t0:.0f}s")
