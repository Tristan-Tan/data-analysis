# -*- coding: utf-8 -*-
"""交付包自检：在跑任何东西之前，先确认本赛段的产物齐不齐、口径对不对

评审人员拿到材料后建议**第一个**运行本脚本。它不训练、不预测，只做静态
校验，几秒钟出结果，能把"缺模型/列顺序不一致/提交文件行序错位"这类问题
在花 3 小时之前就暴露出来。

用法:
    FRAUD_ROUND=A python code/selftest/verify_package.py
    FRAUD_ROUND=B python code/selftest/verify_package.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (P, M, MODEL_DIR, INTERIM, PRED_DIR, TRAIN_DIR, TESTA_DIR,
                    TESTB_DIR, N_FOLDS, FINAL_RECIPE, NOSIL_SEED, TOP_FRAC,
                    ROUND, SPEC, TARGET)
from account_balance import THIN_COLS

FAIL = []
WARN = []


def ok(cond, msg, detail="", warn_only=False):
    tag = "PASS" if cond else ("WARN" if warn_only else "FAIL")
    print(f"  [{tag}] {msg}" + (f"   {detail}" if detail else ""))
    if not cond:
        (WARN if warn_only else FAIL).append(msg)
    return cond


print("=" * 78)
print(f"交付包自检   赛段 {ROUND}   目标测试集 {TARGET}")
print(f"  中间产物 {INTERIM}")
print(f"  模型目录 {MODEL_DIR}")
print(f"  官方成绩 {SPEC['score']}   提交文件 {SPEC['out_name']}")
print("=" * 78)

# ---------- 1. 原始数据 ----------
print("\n[1] 原始数据")
need_raw = [os.path.join(TRAIN_DIR, f) for f in (
    "txn_info_train_testa.csv", "card_info_train_testa.csv",
    "cst_info_train_testa.csv", "accno_info_train_testa.csv", "train.csv")]
need_raw.append(os.path.join(TESTA_DIR, "testa.csv"))
if SPEC["include_testb"]:
    need_raw += [os.path.join(TESTB_DIR, f) for f in (
        "testb.csv", "txn_info_testb.csv", "card_info_testb.csv",
        "cst_info_testb.csv", "accno_info_testb.csv")]
for f in need_raw:
    ok(os.path.exists(f), f"存在 {os.path.relpath(f, os.path.dirname(TRAIN_DIR))}")
if not SPEC["include_testb"] and os.path.exists(os.path.join(TESTB_DIR, "testb.csv")):
    print("  [INFO] 磁盘上有 testB 数据；A 榜链路一律不读取，不影响本赛段")

# ---------- 2. 中间产物 ----------
print("\n[2] 中间产物")
need_interim = ["txn.parquet", "card.parquet", "cst.parquet", "accno.parquet",
                "train.parquet", f"{TARGET}.parquet", "features_v4.parquet",
                "inflow_feat.parquet", "silence_feat.parquet",
                "exp5_tr.parquet"]
pfx = "exp5_te" if TARGET == "testa" else "exp5_teb"
need_interim += [f"{pfx}_f{k}.parquet" for k in range(N_FOLDS)]
if SPEC["use_smy_te"]:
    spfx = "smy_te_te" if TARGET == "testa" else "smy_te_teb"
    need_interim += ["smy_te_tr.parquet"] + \
        [f"{spfx}_f{k}.parquet" for k in range(N_FOLDS)]
missing = [f for f in need_interim if not os.path.exists(P(f))]
ok(not missing, f"{len(need_interim)} 个中间产物齐备",
   "" if not missing else f"缺 {missing[:4]}{' ...' if len(missing) > 4 else ''}")
# A 榜的 interim 里不应出现 testb.parquet
if not SPEC["include_testb"]:
    ok(not os.path.exists(P("testb.parquet")),
       "A 榜中间产物里不含 testb.parquet（赛段隔离生效）")

# ---------- 3. 列顺序 ----------
print("\n[3] 特征列顺序")
base_cols = thin_cols = None
if ok(os.path.exists(M("feature_cols.json")), "存在 feature_cols.json"):
    base_cols = json.load(open(M("feature_cols.json")))
    print(f"         基础维度 = {len(base_cols)}")
    ok("smy_te" in base_cols if SPEC["use_smy_te"]
       else "smy_te" not in base_cols,
       f"smy_te 维度{'存在' if SPEC['use_smy_te'] else '不存在'}，与赛段规格一致")
    ok(len(set(base_cols)) == len(base_cols), "基础列名无重复")
if ok(os.path.exists(M("account_balance_thin_feature_cols.json")),
      "存在 account_balance_thin_feature_cols.json"):
    thin_cols = json.load(open(M("account_balance_thin_feature_cols.json")))
    print(f"         AB-thin 维度 = {len(thin_cols)}")
    if base_cols is not None:
        ok(thin_cols == base_cols + THIN_COLS,
           "AB-thin 列 == 基础列 + 7 维 AB-thin（顺序一致）")

# ---------- 4. 模型文件 ----------
print("\n[4] 模型文件（按本赛段融合配方逐个核对）")
print(f"         配方（重复即权重）: {SPEC['blend']}")
need_models = []
for fam in set(SPEC["blend"]):
    if fam == "dart":
        need_models += [f"dart_s{s}_f{k}.txt"
                        for s in FINAL_RECIPE["dart"] for k in range(N_FOLDS)]
    elif fam == "abthin_lgbn":
        need_models += [f"abthin_lgbn_s{s}_f{k}.txt"
                        for s in FINAL_RECIPE["lgbn"] for k in range(N_FOLDS)]
    elif fam == "cat":
        need_models += [f"cat_s{s}_f{k}.cbm"
                        for s in FINAL_RECIPE["cat"] for k in range(N_FOLDS)]
    elif fam.startswith("nosil_"):
        need_models += [f"{fam}_s{NOSIL_SEED}_f{k}.txt" for k in range(N_FOLDS)]
        j = M(f"{fam}_feature_cols.json")
        if ok(os.path.exists(j), f"存在 {fam}_feature_cols.json"):
            nc = json.load(open(j))
            print(f"         {fam} 维度 = {len(nc)}")
            if base_cols is not None:
                ok(set(nc) <= set(base_cols),
                   f"{fam} 的列是基础列的子集（特征版本一致）")
                ok(nc == [c for c in base_cols if c in set(nc)],
                   f"{fam} 的列顺序与基础列顺序一致")
miss_m = [f for f in need_models if not os.path.exists(M(f))]
ok(not miss_m, f"融合配方所需的 {len(need_models)} 个模型齐备",
   "" if not miss_m else f"缺 {miss_m[:4]}{' ...' if len(miss_m) > 4 else ''}")
extra = [f for f in os.listdir(MODEL_DIR)
         if f.endswith((".txt", ".cbm")) and f not in set(need_models)]
if extra:
    print(f"  [INFO] 目录中另有 {len(extra)} 个不属于本配方的模型文件"
          f"（不影响复现）: {extra[:3]}{' ...' if len(extra) > 3 else ''}")

# ---------- 5. 提交文件 ----------
print("\n[5] 提交文件")
out = os.path.join(PRED_DIR, SPEC["out_name"])
if ok(os.path.exists(out), f"存在 {SPEC['out_name']}", warn_only=True):
    sub = pd.read_csv(out, dtype={"card_no": str})
    ok(list(sub.columns) == ["card_no", "score"], "列为 card_no,score 两列",
       str(list(sub.columns)))
    raw = os.path.join(TESTA_DIR, "testa.csv") if TARGET == "testa" \
        else os.path.join(TESTB_DIR, "testb.csv")
    if os.path.exists(raw):
        rc = pd.read_csv(raw, dtype={"card_no": str})
        ok(len(sub) == len(rc), "行数与原始测试集一致",
           f"{len(sub)} vs {len(rc)}")
        ok((sub["card_no"].values == rc["card_no"].values).all(),
           "卡号顺序与原始测试集**逐行一致**")
    ok(sub["score"].notna().all(), "score 无缺失")
    ok(sub["score"].nunique() > len(sub) * 0.5,
       "score 取值足够分散（不是常数或大面积并列）",
       f"unique={sub['score'].nunique()}/{len(sub)}")
    k = int(len(sub) * TOP_FRAC)
    print(f"         判正张数 top{TOP_FRAC:.0%} = {k}")

print("\n" + "=" * 78)
if FAIL:
    print(f"自检未通过：{len(FAIL)} 项失败")
    for m in FAIL:
        print(f"  - {m}")
    sys.exit(1)
print(f"自检全部通过{f'（{len(WARN)} 项警告）' if WARN else ''}")
for m in WARN:
    print(f"  ! {m}")
