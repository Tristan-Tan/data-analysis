# -*- coding: utf-8 -*-
"""把本包产出的提交文件与「当初实际提交过的那一份」逐卡比对

用途：重组代码之后，最有力的验证不是自检脚本，而是
**新链路产出的 top754 是否与当初拿到成绩的那份文件完全一致**。
若 754/754 全中，说明整条链路（特征 -> 模型 -> 融合配方 -> 硬规则 -> 回填）
与当初逐环节等价；若有差异，差多少张、差在哪个排名带，这里也一并给出。

评分只看前 1% 是哪些卡，score 的具体数值与顺序不影响得分，
因此比对的是**集合**而非数值。

用法:
    FRAUD_ROUND=B python code/selftest/compare_submission.py \\
        prediction_result/prediction_resultB_fuse_partial_x2plus_full.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PRED_DIR, TOP_FRAC, SPEC, ROUND

if len(sys.argv) < 2:
    raise SystemExit("用法: python code/selftest/compare_submission.py <参照csv路径>")
ref_path = sys.argv[1]
new_path = os.path.join(PRED_DIR, SPEC["out_name"])
for p in (ref_path, new_path):
    if not os.path.exists(p):
        raise SystemExit(f"文件不存在: {p}")


def load(p):
    d = pd.read_csv(p, dtype={"card_no": str})
    assert {"card_no", "score"} <= set(d.columns), f"{p} 缺少 card_no/score"
    return d


ref, new = load(ref_path), load(new_path)
print("=" * 76)
print(f"赛段 {ROUND}   官方成绩 {SPEC['score']}")
print(f"  参照（当初实际提交）: {os.path.basename(ref_path)}  {ref.shape}")
print(f"  本次（新链路产出）  : {os.path.basename(new_path)}  {new.shape}")
print("=" * 76)

assert len(ref) == len(new), f"行数不一致 {len(ref)} vs {len(new)}"
assert set(ref["card_no"]) == set(new["card_no"]), "卡号集合不一致"
same_order = (ref["card_no"].values == new["card_no"].values).all()
print(f"\n卡号逐行顺序一致: {same_order}"
      + ("" if same_order else "（不影响得分，但提交文件应与测试集同序）"))

K = int(len(new) * TOP_FRAC)
tr = set(ref.nlargest(K, "score")["card_no"])
tn = set(new.nlargest(K, "score")["card_no"])
inter = len(tr & tn)
print(f"\ntop{K} 集合重叠: {inter}/{K}   差异 {K - inter} 张")

# ---- 整体排序是否一致：把"边界抖动"与"链路有实质差异"定量区分开 ----
m = ref[["card_no", "score"]].rename(columns={"score": "s_ref"}).merge(
    new[["card_no", "score"]].rename(columns={"score": "s_new"}), on="card_no")
assert len(m) == len(ref), "按卡号对齐后行数异常"
r_ref = rankdata(-m["s_ref"].values, method="ordinal")
r_new = rankdata(-m["s_new"].values, method="ordinal")
rho = spearmanr(r_ref, r_new).correlation
disp = np.abs(r_ref - r_new)
print(f"\n【整体排序一致性】")
print(f"  全量 {len(m):,} 张卡的 Spearman: {rho:.6f}")
print(f"  名次位移 |Δrank|: p50={int(np.median(disp))} "
      f"p99={int(np.percentile(disp, 99))} max={int(disp.max())}")
print(f"  位移 > {K} 名的卡: {int((disp > K).sum())} 张"
      f"（占 {(disp > K).mean():.4%}）")

# ---- 边界有多密：分数几乎并列时，谁进 top-K 本就由浮点末位决定 ----
srt = np.sort(m["s_ref"].values)[::-1]
span = srt[0] - srt[-1]
lo, hi = max(K - 60, 0), min(K + 60, len(srt) - 1)
if span > 0:
    print(f"\n【边界密度（参照文件）】")
    ratio = (srt[lo] - srt[hi]) / span
    print(f"  第 {lo}~{hi} 名的分数跨度占全量极差的 {ratio:.4%}")
    if ratio < 0.01:
        print("  => 边界处分数近乎并列，谁进 top-K 由浮点末位决定，"
              "\n     差异卡集中在边界属 best_iteration 抖动的正常表现")
    else:
        print("  => 边界分数并不密集，差异卡需要更谨慎看待，"
              "\n     应结合上面的 Spearman 与名次位移一并判断")

if inter == K:
    print("\n  ✔ 完全一致。新链路与当初取得该成绩的链路逐环节等价，")
    print("    可以认定本包能精确复现该成绩。")
else:
    # 差异卡落在什么排名带，用于判断是不是 best_iteration 的边界抖动
    rr = pd.Series(rankdata(-ref["score"].values, method="ordinal"),
                   index=ref["card_no"]).to_dict()
    rn = pd.Series(rankdata(-new["score"].values, method="ordinal"),
                   index=new["card_no"]).to_dict()
    out_only = sorted(tr - tn, key=lambda c: rr[c])
    in_only = sorted(tn - tr, key=lambda c: rn[c])
    print(f"\n  参照有、本次无 {len(out_only)} 张 —— 它们在本次的新排名:")
    q = np.array([rn[c] for c in out_only])
    print(f"    min={q.min()} p50={int(np.median(q))} max={q.max()}")
    print(f"  本次有、参照无 {len(in_only)} 张 —— 它们在参照里的原排名:")
    q2 = np.array([rr[c] for c in in_only])
    print(f"    min={q2.min()} p50={int(np.median(q2))} max={q2.max()}")
    near = int(((q <= K * 1.2) & (q > K)).sum())
    print(f"\n  判读：若差异卡的排名都紧贴 {K} 边界（本次 {near}/{len(out_only)}"
          f" 张落在 {K}~{int(K*1.2)} 名），属 best_iteration 浮点抖动，")
    print("        与说明文档 §四（七）预估的 ±3 卡同性质；")
    print("        若差异卡散布在很靠后的排名，说明链路存在实质差异，需排查。")
print()
