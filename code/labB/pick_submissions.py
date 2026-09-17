# -*- coding: utf-8 -*-
"""提交前查重与选点：在候选 csv 里挑彼此最不相同的几个

【为什么需要】prediction_result/ 下有 20+ 个候选，命名高度相似
（例如 fuse_plus_partial 与 nosil4fam_partial 的 top754 重叠都是 747/754）。
若两份文件的 **top754 集合完全相同**，提交它们等于浪费一次机会——
评分只看前 754 是哪些卡，score 的具体数值与顺序都不影响得分。

【本脚本做什么】只读，不写，不提交。
  1. 列出所有候选的 top754 与"当前已提交版本"的重叠
  2. 标出**完全重复**的文件组（top754 集合一模一样）
  3. 在未提交的候选里做两两重叠矩阵，按"彼此最不相同"给出选点建议
  4. 附上记录中已实测的 OOF 差值（来自 B榜优化过程记录.md §四.2），
     用于剔除 OOF 明显退化的候选

【期望值提醒】历史实测：五族版换动 30 张、线上不变（663→663）；
组内分位修复版换动极少、线上 −1。边界处换卡是轻微负期望，
本脚本的作用是**避免浪费**，不是提高上限。

用法: PYTHONPATH=code python code/labB/pick_submissions.py testb
耗时: 约 10~30 秒
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P, PRED_DIR, TOP_FRAC
from lab_common import LAB_PRED

TARGET = sys.argv[1] if len(sys.argv) > 1 else "testb"

# 已提交过的文件（来自 B榜优化过程记录.md §一 成绩时间线）
SUBMITTED = {
    "prediction_resultB.csv": "0.87931 (663) 三族融合",
    "prediction_resultB_fuse_plus_both_5fam.csv": "0.87931 (663) 五族",
}
# 记录中已实测的 OOF 差值（§四.2 融合结构枚举，不重训）
KNOWN_OOF = {
    "prediction_resultB_fuse_plus_partial.csv": +3,
    "prediction_resultB_nosil4fam_partial.csv": +3,
    "prediction_resultB_fuse_partial_x3.csv": -2,
    "prediction_resultB_fuse_plus_both_5fam.csv": -3,
    "prediction_resultB_fuse_plus_full.csv": -18,
    "prediction_resultB_nosil4fam.csv": -18,
    "prediction_resultB_fuse_full_x2.csv": -35,
    "prediction_resultB_fuse_full_x3.csv": -43,
}
# 机制上已被判死，不参与选点（§四.3 图平滑；§五.8 r01）
BANNED_SUBSTR = ("_gs_deg", "r01_")

te = pd.read_parquet(P(f"{TARGET}.parquet"))
te["card_no"] = te["card_no"].astype(str)
K = int(len(te) * TOP_FRAC)
CARDS = set(te["card_no"])
print(f"{TARGET} {len(te):,} 张，top1% = {K}\n")

files = sorted(glob.glob(os.path.join(PRED_DIR, "*.csv"))) + \
    sorted(glob.glob(os.path.join(LAB_PRED, "*.csv")))
tops, bad = {}, []
for f in files:
    name = os.path.relpath(f, PRED_DIR)
    try:
        c = pd.read_csv(f, dtype={"card_no": str})
        if "score" not in c.columns or len(c) != len(te) \
                or set(c["card_no"]) != CARDS:
            bad.append((name, "卡号集合或行数与 testB 不符"))
            continue
        tops[name] = frozenset(c.nlargest(K, "score")["card_no"])
    except Exception as e:                                   # noqa: BLE001
        bad.append((name, f"{type(e).__name__}: {e}"))
for n, why in bad:
    print(f"  跳过 {n}: {why}")

REF = "prediction_resultB.csv"
assert REF in tops, f"找不到基准文件 {REF}"
ref = tops[REF]

# ---------------- 1. 完全重复分组 ----------------
print("\n" + "=" * 84)
print("【1】top754 集合完全相同的文件组（组内只值得提交一个）")
print("=" * 84)
groups = {}
for n, t in tops.items():
    groups.setdefault(t, []).append(n)
dups = [g for g in groups.values() if len(g) > 1]
if dups:
    for i, g in enumerate(dups, 1):
        print(f"  组{i}: " + "  ==  ".join(g))
else:
    print("  没有完全重复的文件")

# ---------------- 2. 与当前已提交版本的距离 ----------------
print("\n" + "=" * 84)
print(f"【2】各候选与基准 {REF} 的差异（换动 = 754 - 重叠）")
print("=" * 84)
rows = []
for n, t in tops.items():
    if any(b in n for b in BANNED_SUBSTR):
        continue
    rows.append({"文件": n, "重叠": len(t & ref), "换动": K - len(t & ref),
                 "已提交": SUBMITTED.get(os.path.basename(n), ""),
                 "记录OOF差": KNOWN_OOF.get(os.path.basename(n), None)})
d = pd.DataFrame(rows).sort_values("换动")
print(d.to_string(index=False, na_rep="-"))
print(f"\n  （已排除机制上判死的: {BANNED_SUBSTR}）")

# ---------------- 3. 未提交候选的两两重叠 ----------------
print("\n" + "=" * 84)
print("【3】未提交候选之间的两两换动（数值越大越不相同，采样点越分散）")
print("=" * 84)
cand = [n for n in d["文件"]
        if not SUBMITTED.get(os.path.basename(n))
        and tops[n] != ref]
# 完全重复的只留一个
seen, uniq = set(), []
for n in cand:
    if tops[n] in seen:
        continue
    seen.add(tops[n])
    uniq.append(n)
print(f"  去重后剩 {len(uniq)} 个互不相同的候选\n")
if len(uniq) >= 2:
    M = pd.DataFrame(
        [[K - len(tops[a] & tops[b]) for b in uniq] for a in uniq],
        index=[os.path.basename(x)[:40] for x in uniq],
        columns=[os.path.basename(x)[:18] for x in uniq])
    print(M.to_string())

# ---------------- 4. 选点建议 ----------------
print("\n" + "=" * 84)
print("【4】选点建议：OOF 未明显退化，且彼此最分散的 3 个")
print("=" * 84)
ok = [n for n in uniq
      if (KNOWN_OOF.get(os.path.basename(n)) is None
          or KNOWN_OOF[os.path.basename(n)] >= -5)]
print(f"  OOF 未明显退化（>= -5 或无记录）的候选 {len(ok)} 个")
if len(ok) >= 2:
    # 贪心取彼此最远的三个
    pick = [max(ok, key=lambda n: K - len(tops[n] & ref))]
    while len(pick) < min(3, len(ok)):
        rest = [n for n in ok if n not in pick]
        nxt = max(rest, key=lambda n: min(K - len(tops[n] & tops[p])
                                          for p in pick))
        pick.append(nxt)
    for i, n in enumerate(pick, 1):
        oofd = KNOWN_OOF.get(os.path.basename(n))
        print(f"  {i}. {n:<52} 与基准换动 {K-len(tops[n]&ref):>3}  "
              f"记录OOF差 {'-' if oofd is None else f'{oofd:+d}'}")
    print("\n  ⚠ 这是『在已有候选里避免浪费』的选点，不是『提高上限』。")
    print("    历史实测：换动 30 张 -> 线上不变；换动极少 -> 线上 −1。")
    print("    期望 663，乐观 664~665。不要据此调整对结果的预期。")
else:
    print("  可用候选不足，不做建议")
