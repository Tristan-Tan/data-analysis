# -*- coding: utf-8 -*-
"""预测：加载 model/ 下已训练好的模型，输出提交文件

用法:
    python code/test/predict.py            # 预测 testA（默认）
    python code/test/predict.py testb      # 预测 testB（复赛）

流程:
  1. 用与训练完全相同的 assemble 代码构建测试矩阵（5 份 fold-matched）
  2. 第 k 折的模型预测第 k 份矩阵，5 折平均 -> 该 seed 的预测
  3. 同族多 seed 做 rank 平均 -> 族预测
  4. dart(401) / abthin_lgbn(408) / cat(401) 三族 rank 等权平均
  5. 硬规则：sil_days >= 32 的卡置顶（train 中该条件下 8/8 全为欺诈）
  6. 按测试集主键顺序回填，写出 prediction_result/prediction_resultA.csv

耗时: 约 3 分钟（不训练）
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P, MODEL_DIR, PRED_DIR, N_FOLDS, FINAL_RECIPE
from assemble import build_test_folds
from account_balance import (THIN_COLS, append_features,
                             load_or_create_features)

t0 = time.time()
TARGET = sys.argv[1] if len(sys.argv) > 1 else "testa"
OUT_NAME = "prediction_resultA.csv" if TARGET == "testa" else "prediction_resultB.csv"


def R(v):
    return rankdata(v) / len(v)


def predict_family(Xte, cols, family, seeds):
    acc = np.zeros(len(Xte[0]))
    for sd in seeds:
        pte = np.zeros(len(Xte[0]))
        for k in range(N_FOLDS):
            if family == "cat":
                p = os.path.join(MODEL_DIR, f"cat_s{sd}_f{k}.cbm")
                m = CatBoostClassifier()
                m.load_model(p)
                pte += m.predict_proba(Xte[k][cols])[:, 1] / N_FOLDS
            else:
                p = os.path.join(MODEL_DIR, f"{family}_s{sd}_f{k}.txt")
                m = lgb.Booster(model_file=p)
                pte += m.predict(Xte[k][cols]) / N_FOLDS
        acc += R(pte)
        print(f"  [{family}] seed={sd} 完成  {time.time()-t0:.0f}s")
    return acc / len(seeds)


def main():
    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        base_cols = json.load(f)
    with open(os.path.join(
            MODEL_DIR, "account_balance_thin_feature_cols.json")) as f:
        thin_cols = json.load(f)
    assert thin_cols == base_cols + THIN_COLS, "AB-thin 特征列或顺序不一致"

    te, Xte_base = build_test_folds(TARGET, N_FOLDS, base_cols)
    ab = load_or_create_features()
    Xte_thin = [append_features(x, te[["card_no"]], ab)
                for x in Xte_base]
    print(
        f"测试矩阵 base={Xte_base[0].shape} AB-thin={Xte_thin[0].shape} "
        f"× {N_FOLDS} 份   {time.time()-t0:.0f}s"
    )

    fam = {
        "dart": predict_family(
            Xte_base, base_cols, "dart", FINAL_RECIPE["dart"]),
        "abthin_lgbn": predict_family(
            Xte_thin, thin_cols, "abthin_lgbn", FINAL_RECIPE["lgbn"]),
        "cat": predict_family(
            Xte_base, base_cols, "cat", FINAL_RECIPE["cat"]),
    }

    score = R(np.mean([
        R(fam["dart"]), R(fam["abthin_lgbn"]), R(fam["cat"])
    ], axis=0))

    # 硬规则：数据抽取伪影 —— 正常卡末笔必在窗口起点之后
    sil = pd.read_parquet(P("silence_feat.parquet"),
                          columns=["card_no", "sil_days"])
    sil["card_no"] = sil["card_no"].astype(str)
    sv = te.merge(sil, on="card_no", how="left")["sil_days"].values
    n_hard = int((sv >= 32).sum())
    if n_hard:
        score[sv >= 32] = 1.0
    print(f"硬规则命中 {n_hard} 张（sil_days >= 32）")

    # 按测试集主键顺序回填（硬约束）
    sub = te[["card_no"]].copy()
    sub["score"] = score
    assert len(sub) == len(te)
    assert (sub["card_no"].values == te["card_no"].values).all()
    out = os.path.join(PRED_DIR, OUT_NAME)
    sub.to_csv(out, index=False)
    print(f"已写出 {out}   {sub.shape}   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
