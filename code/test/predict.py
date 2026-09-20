# -*- coding: utf-8 -*-
"""预测：加载 model/{赛段}/ 下已训练好的模型，输出提交文件

用法:
    bash code/scripts/run_predict.sh A      # 初赛 -> prediction_resultA.csv
    bash code/scripts/run_predict.sh B      # 复赛 -> prediction_resultB.csv

**本脚本不训练**，只加载落盘模型，耗时约 3~5 分钟。

流程:
  1. 用与训练完全相同的 assemble 代码构建测试矩阵（5 份 fold-matched）
  2. 第 k 折的模型预测第 k 份矩阵，5 折平均 -> 该 seed 的预测
  3. 同族多 seed 做 rank 平均 -> 族预测
  4. 按 config.SPEC["blend"] 做 rank 等权平均，**列表里重复出现即表示权重**
        A 榜: dart + abthin_lgbn + cat
        B 榜: dart + abthin_lgbn + cat + nosil_partial×2 + nosil_full×1
  5. 硬规则：sil_days >= 32 的卡置顶（train 中该条件下 8/8 全为欺诈）
  6. 按测试集主键顺序回填，保证行数与顺序与原测试集 csv 完全一致

⚠ 模型保存时已按 best_iteration 截断（save_model(num_iteration=...)），
  因此这里不再传迭代数，predict 默认用全部树即为当时的最优迭代。
"""
import os
import sys
import json
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (P, M, MODEL_DIR, PRED_DIR, N_FOLDS, FINAL_RECIPE,
                    NOSIL_SEED, HARD_SIL_DAYS, TOP_FRAC, SPEC, ROUND, banner)
from assemble import build_test_folds
from account_balance import (THIN_COLS, append_features,
                             load_or_create_features)

t0 = time.time()


def R(v):
    return rankdata(v) / len(v)


def _load_cols(fname):
    p = M(fname)
    assert os.path.exists(p), (
        f"缺少列顺序文件 {p}\n请先按 README 运行 run_train.sh {ROUND}")
    with open(p) as f:
        return json.load(f)


def predict_family(Xte, cols, family, seeds, is_cat=False):
    """同族多 seed 的 rank 平均。每个 seed 内部先做 5 折平均。"""
    acc = np.zeros(len(Xte[0]))
    for sd in seeds:
        pte = np.zeros(len(Xte[0]))
        for k in range(N_FOLDS):
            ext = "cbm" if is_cat else "txt"
            p = M(f"{family}_s{sd}_f{k}.{ext}")
            assert os.path.exists(p), f"缺少模型 {p}"
            if is_cat:
                m = CatBoostClassifier()
                m.load_model(p)
                pte += m.predict_proba(Xte[k][cols])[:, 1] / N_FOLDS
            else:
                m = lgb.Booster(model_file=p)
                pte += m.predict(Xte[k][cols]) / N_FOLDS
        acc += R(pte)
        print(f"  [{family}] seed={sd} 完成  {time.time()-t0:.0f}s")
    return acc / len(seeds)


def main():
    banner("predict")
    base_cols = _load_cols("feature_cols.json")
    thin_cols = _load_cols("account_balance_thin_feature_cols.json")
    assert thin_cols == base_cols + THIN_COLS, "AB-thin 特征列或顺序不一致"

    te, Xte_base = build_test_folds(N_FOLDS, base_cols)
    ab = load_or_create_features()
    Xte_thin = [append_features(x, te[["card_no"]], ab) for x in Xte_base]
    print(f"测试矩阵 base={Xte_base[0].shape} AB-thin={Xte_thin[0].shape}"
          f" × {N_FOLDS} 份   {time.time()-t0:.0f}s")

    # 每族只算一次；blend 里重复出现的成员复用缓存，不重复推理
    blend = SPEC["blend"]
    print(f"融合配方（重复即权重）: {blend}")
    fam = OrderedDict()
    for name in blend:
        if name in fam:
            continue
        if name == "dart":
            fam[name] = predict_family(Xte_base, base_cols, "dart",
                                       FINAL_RECIPE["dart"])
        elif name == "abthin_lgbn":
            fam[name] = predict_family(Xte_thin, thin_cols, "abthin_lgbn",
                                       FINAL_RECIPE["lgbn"])
        elif name == "cat":
            fam[name] = predict_family(Xte_base, base_cols, "cat",
                                       FINAL_RECIPE["cat"], is_cat=True)
        elif name.startswith("nosil_"):
            cols = _load_cols(f"{name}_feature_cols.json")
            assert set(cols) <= set(base_cols), \
                f"{name} 的列不是基础列的子集，特征版本不一致"
            fam[name] = predict_family(Xte_base, cols, name, [NOSIL_SEED])
        else:
            raise ValueError(f"未知的融合成员 {name}")

    score = R(np.mean([R(fam[n]) for n in blend], axis=0))

    # 硬规则：数据抽取伪影 —— 正常卡末笔必在窗口起点之后
    sil = pd.read_parquet(P("silence_feat.parquet"),
                          columns=["card_no", "sil_days"])
    sil["card_no"] = sil["card_no"].astype(str)
    sv = te.merge(sil, on="card_no", how="left")["sil_days"].values
    n_hard = int((sv >= HARD_SIL_DAYS).sum())
    if n_hard:
        score[sv >= HARD_SIL_DAYS] = 1.0
    print(f"硬规则命中 {n_hard} 张（sil_days >= {HARD_SIL_DAYS}）")

    # 按测试集主键顺序回填（硬约束）
    sub = te[["card_no"]].copy()
    sub["score"] = score
    assert len(sub) == len(te)
    assert (sub["card_no"].values == te["card_no"].values).all()
    out = os.path.join(PRED_DIR, SPEC["out_name"])
    sub.to_csv(out, index=False)
    k = int(len(sub) * TOP_FRAC)
    print(f"\n已写出 {out}   {sub.shape}")
    print(f"  判正张数 top{TOP_FRAC:.0%} = {k}；本赛段官方成绩 {SPEC['score']}")
    print(f"  完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
