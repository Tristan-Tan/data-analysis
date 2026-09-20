# -*- coding: utf-8 -*-
"""端到端自检：在合成数据上把 A / B 两个赛段的全链路各跑一遍并断言关键性质

⚠ 这只验证**流水线正确性**（赛段隔离、列顺序、模型落盘与加载、融合配方、
   提交文件格式），**不代表任何效果验证** —— 合成数据没有真实信号。

跑完 A、B 两个赛段后断言：
  ISO-1  A 榜中间产物里没有 testb.parquet（testB 一律不读）
  ISO-2  A 榜特征集里没有只出现在 testB 的摘要码，B 榜里有
         —— 这是"参考池不同则**特征集本身**不同"的直接证据
  ISO-3  A 榜列里没有 smy_te，B 榜里有
  ISO-4  两榜基础维度不同，且两榜的 interim/model 目录互不覆盖
  COL-1  AB-thin 列 == 基础列 + 7 维，顺序一致
  COL-2  nosil 两族的列是基础列的**子集**且保持基础列顺序
  MDL-1  A 榜 20 个模型、B 榜 30 个模型齐备
  RPD-1  predict 的输出与训练时落盘的 te_*.npy **排序完全一致**
         —— 这是"路径一复现误差为 0"这一声明的直接验证
  SUB-1  提交文件行数与逐行卡号顺序与原始测试集一致
  SUB-2  重跑一次 predict，输出逐字节相同（确定性）

用法（需先准备好合成数据）:
    python code/selftest/make_synth_data.py /path/to/sandbox
    FRAUD_BASE=/path/to/sandbox bash code/scripts/run_train.sh A
    FRAUD_BASE=/path/to/sandbox bash code/scripts/run_predict.sh A
    FRAUD_BASE=/path/to/sandbox bash code/scripts/run_train.sh B
    FRAUD_BASE=/path/to/sandbox bash code/scripts/run_predict.sh B
    python code/selftest/e2e_test.py /path/to/sandbox
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
BASE = sys.argv[1] if len(sys.argv) > 1 else "."
ok_all = True


def chk(name, cond, detail=""):
    global ok_all
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    ok_all = ok_all and bool(cond)


def interim(r, n):
    return os.path.join(BASE, "data", f"interim_{r}", n)


def model(r, n):
    return os.path.join(BASE, "model", r, n)


def cols(r, n="feature_cols.json"):
    with open(model(r, n)) as f:
        return json.load(f)


print("=" * 78)
print(f"端到端自检（合成数据，不代表效果验证）  沙盒 {BASE}")
print("=" * 78)

print("\n[ISO] 赛段隔离")
chk("ISO-1 A 榜中间产物里没有 testb.parquet",
    not os.path.exists(interim("A", "testb.parquet")))
chk("ISO-1 B 榜中间产物里有 testb.parquet",
    os.path.exists(interim("B", "testb.parquet")))

ca, cb = cols("A"), cols("B")
only_b = [c for c in ("smy2_9201", "smy2_9202")]
chk("ISO-2 A 榜特征集不含只出现在 testB 的摘要码",
    all(c not in ca for c in only_b),
    f"A 中命中 {[c for c in only_b if c in ca]}")
chk("ISO-2 B 榜特征集含这些摘要码（参考池不同 -> 特征集不同）",
    all(c in cb for c in only_b),
    f"B 中命中 {[c for c in only_b if c in cb]}")
chk("ISO-3 A 榜不含 smy_te", "smy_te" not in ca)
chk("ISO-3 B 榜含 smy_te", "smy_te" in cb)
chk("ISO-4 两榜基础维度不同", len(ca) != len(cb), f"A={len(ca)} B={len(cb)}")
chk("ISO-4 txn.parquet 两榜大小不同（B 含 testB 流水）",
    os.path.getsize(interim("A", "txn.parquet"))
    != os.path.getsize(interim("B", "txn.parquet")))

print("\n[COL] 列顺序一致性")
sys.path.insert(0, os.path.join(ROOT, "code"))
os.environ.setdefault("FRAUD_ROUND", "B")
os.environ["FRAUD_BASE"] = BASE
from account_balance import THIN_COLS                            # noqa: E402
for r, c in (("A", ca), ("B", cb)):
    t = cols(r, "account_balance_thin_feature_cols.json")
    chk(f"COL-1 {r} 榜 AB-thin == 基础列 + {len(THIN_COLS)} 维且顺序一致",
        t == c + THIN_COLS, f"{len(c)} + {len(THIN_COLS)} -> {len(t)}")
for mode in ("partial", "full"):
    nc = cols("B", f"nosil_{mode}_feature_cols.json")
    chk(f"COL-2 nosil_{mode} 是基础列的子集", set(nc) <= set(cb))
    chk(f"COL-2 nosil_{mode} 保持基础列顺序",
        nc == [c for c in cb if c in set(nc)],
        f"{len(cb)} -> {len(nc)}")

print("\n[MDL] 模型文件")
na = len([f for f in os.listdir(os.path.join(BASE, "model", "A"))
          if f.endswith((".txt", ".cbm"))])
nb = len([f for f in os.listdir(os.path.join(BASE, "model", "B"))
          if f.endswith((".txt", ".cbm"))])
chk("MDL-1 A 榜 20 个模型", na == 20, f"实得 {na}")
chk("MDL-1 B 榜 30 个模型（20 + nosil 两族各 5）", nb == 30, f"实得 {nb}")

print("\n[RPD] 落盘模型的预测 == 训练时的预测（路径一误差为 0 的直接验证）")


def R(v):
    return rankdata(v) / len(v)


BLEND = {"A": ["dart", "abthin_lgbn", "cat"],
         "B": ["dart", "abthin_lgbn", "cat",
               "nosil_partial", "nosil_partial", "nosil_full"]}
for r in ("A", "B"):
    tgt = "testa" if r == "A" else "testb"
    parts = []
    missing = False
    for m in BLEND[r]:
        p = interim(r, f"te_{m}.npy")
        if not os.path.exists(p):
            missing = True
            break
        parts.append(R(np.load(p)))
    if missing:
        chk(f"RPD-1 {r} 榜 te_*.npy 齐备", False, f"缺 te_{m}.npy")
        continue
    expect = R(np.mean(parts, axis=0))
    # 硬规则：与 predict.py 完全一致
    te = pd.read_parquet(interim(r, f"{tgt}.parquet"))
    te["card_no"] = te["card_no"].astype(str)
    sil = pd.read_parquet(interim(r, "silence_feat.parquet"),
                          columns=["card_no", "sil_days"])
    sil["card_no"] = sil["card_no"].astype(str)
    sv = te.merge(sil, on="card_no", how="left")["sil_days"].values
    expect = expect.copy()
    expect[sv >= 32] = 1.0
    out = os.path.join(BASE, "prediction_result",
                       f"prediction_result{r}.csv")
    got = pd.read_csv(out, dtype={"card_no": str})["score"].values
    same = np.array_equal(rankdata(expect, method="ordinal"),
                          rankdata(got, method="ordinal"))
    chk(f"RPD-1 {r} 榜 predict 输出与训练侧 te_*.npy 排序完全一致", same)

print("\n[SUB] 提交文件")
for r in ("A", "B"):
    tgt = "testa" if r == "A" else "testb"
    raw = os.path.join(BASE, "data", "testA" if r == "A" else "testB",
                       f"{tgt}.csv")
    out = os.path.join(BASE, "prediction_result",
                       f"prediction_result{r}.csv")
    sub = pd.read_csv(out, dtype={"card_no": str})
    rc = pd.read_csv(raw, dtype={"card_no": str})
    chk(f"SUB-1 {r} 榜列为 card_no,score", list(sub.columns) == ["card_no", "score"])
    chk(f"SUB-1 {r} 榜行数与原始测试集一致", len(sub) == len(rc),
        f"{len(sub)} vs {len(rc)}")
    chk(f"SUB-1 {r} 榜卡号逐行顺序一致",
        (sub["card_no"].values == rc["card_no"].values).all())

print("\n[SUB-2] 预测确定性（重跑一次 predict，比对逐字节）")
for r in ("A", "B"):
    out = os.path.join(BASE, "prediction_result", f"prediction_result{r}.csv")
    before = open(out, "rb").read()
    env = dict(os.environ, FRAUD_BASE=BASE, FRAUD_ROUND=r,
               PYTHONPATH=os.path.join(ROOT, "code"))
    p = subprocess.run([sys.executable, "code/test/predict.py"],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        chk(f"SUB-2 {r} 榜 predict 重跑成功", False, p.stderr[-400:])
        continue
    chk(f"SUB-2 {r} 榜重跑输出逐字节相同",
        open(out, "rb").read() == before)

print("\n" + "=" * 78)
print("端到端自检全部通过（仅验证流水线正确性，不代表效果验证）" if ok_all
      else "存在失败用例，请勿交付")
sys.exit(0 if ok_all else 1)
