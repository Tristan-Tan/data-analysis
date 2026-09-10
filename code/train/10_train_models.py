# -*- coding: utf-8 -*-
"""10 训练最终模型并落盘

最终提交 = 三个模型族的 rank 等权平均：
    dart          402 维 LightGBM DART                  seed [42]
    abthin_lgbn   409 维 LightGBM 窄树（402 + AB-thin） seed [42, 2026]
    cat           402 维 CatBoost 对称树                seed [42]

⚠ 402 维 = A 榜提交时的 401 维 + 07b 新增的 smy_cd 目标编码（1 维）。
   若要精确复现已提交的 A 榜 0.916446（401/408 维），
   请勿运行 07b_smy_te_encoding.py，直接跳过该步骤即可退回原始维度。

为什么是这三个而不是「线下最好的那个」：见说明文档 §一.6。
简言之，线下 OOF 分数经过数十轮选择后已带过拟合，2772~2780 的线下差异
在线上只对应随机摆动；而成员多、结构异构的组合，选择性过拟合被平均得最彻底。

⚠ 折划分固定 random_state=42 —— exp5 是在该折下 fold-safe 生成的。
⚠ 第 k 折的模型使用第 k 份 fold-matched 测试编码做预测。

输出:
  model/{family}_s{seed}_f{fold}.txt|cbm   共 20 个模型
  model/feature_cols.json                  402 维基础列顺序
  model/account_balance_thin_feature_cols.json  409 维 AB-thin 列顺序
  data/interim/oof_{family}.npy            OOF（供线下核对）
耗时: 约 110 分钟（dart 约 78 分钟，是主要开销）
"""
import os
import sys
import json
import time

import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (P, MODEL_DIR, FOLD_SEED, N_FOLDS, FINAL_RECIPE,
                    LGB_BASE, DART_EXTRA, LGBN_EXTRA, CAT_PARAMS, top1_f1)
from assemble import build_train, build_test_folds
from account_balance import (THIN_COLS, append_features,
                             load_or_create_features)

t0 = time.time()


def train_lgb_family(X, y, Xte, cols, folds, family, extra, seeds):
    oof_acc = np.zeros(len(X))
    te_acc = np.zeros(len(Xte[0]))
    for sd in seeds:
        p = dict(LGB_BASE, seed=sd)
        p.update(extra)
        oof = np.zeros(len(X))
        pte = np.zeros(len(Xte[0]))
        for k, (ti, vi) in enumerate(folds):
            m = lgb.train(p, lgb.Dataset(X.iloc[ti][cols], y[ti]), 3000,
                          valid_sets=[lgb.Dataset(X.iloc[vi][cols], y[vi])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            m.save_model(os.path.join(MODEL_DIR, f"{family}_s{sd}_f{k}.txt"),
                         num_iteration=m.best_iteration)
            oof[vi] = m.predict(X.iloc[vi][cols], num_iteration=m.best_iteration)
            pte += m.predict(Xte[k][cols], num_iteration=m.best_iteration) / N_FOLDS
        _, tp = top1_f1(y, oof)
        print(f"  [{family}] seed={sd:<6} OOF {tp}/3000   {time.time()-t0:.0f}s")
        oof_acc += rankdata(oof) / len(oof)
        te_acc += rankdata(pte) / len(pte)
    return oof_acc / len(seeds), te_acc / len(seeds)


def train_cat_family(X, y, Xte, cols, folds, seeds):
    oof_acc = np.zeros(len(X))
    te_acc = np.zeros(len(Xte[0]))
    for sd in seeds:
        oof = np.zeros(len(X))
        pte = np.zeros(len(Xte[0]))
        for k, (ti, vi) in enumerate(folds):
            m = CatBoostClassifier(random_seed=sd, **CAT_PARAMS)
            m.fit(X.iloc[ti][cols], y[ti],
                  eval_set=(X.iloc[vi][cols], y[vi]), use_best_model=True)
            m.save_model(os.path.join(MODEL_DIR, f"cat_s{sd}_f{k}.cbm"))
            oof[vi] = m.predict_proba(X.iloc[vi][cols])[:, 1]
            pte += m.predict_proba(Xte[k][cols])[:, 1] / N_FOLDS
        _, tp = top1_f1(y, oof)
        print(f"  [cat] seed={sd:<6} OOF {tp}/3000   {time.time()-t0:.0f}s")
        oof_acc += rankdata(oof) / len(oof)
        te_acc += rankdata(pte) / len(pte)
    return oof_acc / len(seeds), te_acc / len(seeds)


def main():
    tr, Xbase, y, base_cols = build_train()
    print(f"基础训练矩阵 {Xbase.shape}   {time.time()-t0:.0f}s")

    # 财富归一自检（口径错了这里立刻会露出来）
    m1 = Xbase.loc[y == 1, "in_vs_aum"].mean()
    m0 = Xbase.loc[y == 0, "in_vs_aum"].mean()
    print(f"自检 in_vs_aum: label1={m1:.2f} label0={m0:.2f}  （应约 178 / 21）")

    te, Xte_base = build_test_folds("testa", N_FOLDS, base_cols)
    print(f"基础测试矩阵 {Xte_base[0].shape} × {N_FOLDS} 份（fold-matched）")

    ab = load_or_create_features()
    Xthin = append_features(Xbase, tr[["card_no"]], ab)
    Xte_thin = [append_features(x, te[["card_no"]], ab)
                for x in Xte_base]
    thin_cols = base_cols + THIN_COLS
    assert Xthin.columns.tolist() == thin_cols
    print(f"AB-thin 训练/测试矩阵 {Xthin.shape}/{Xte_thin[0].shape}")

    with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w") as f:
        json.dump(base_cols, f, ensure_ascii=False, indent=2)
    with open(os.path.join(
            MODEL_DIR, "account_balance_thin_feature_cols.json"), "w") as f:
        json.dump(thin_cols, f, ensure_ascii=False, indent=2)

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True,
                                 random_state=FOLD_SEED).split(Xbase, y))

    print("\n=== abthin_lgbn（402 + 7 维账户/余额薄特征）===")
    o, t = train_lgb_family(Xthin, y, Xte_thin, thin_cols, folds,
                            "abthin_lgbn",
                            LGBN_EXTRA, FINAL_RECIPE["lgbn"])
    np.save(P("oof_abthin_lgbn.npy"), o)
    np.save(P("te_abthin_lgbn.npy"), t)

    print("\n=== cat（CatBoost 对称树）===")
    o, t = train_cat_family(
        Xbase, y, Xte_base, base_cols, folds, FINAL_RECIPE["cat"])
    np.save(P("oof_cat.npy"), o); np.save(P("te_cat.npy"), t)

    print("\n=== dart（LightGBM DART，耗时最长）===")
    o, t = train_lgb_family(Xbase, y, Xte_base, base_cols, folds, "dart",
                            DART_EXTRA, FINAL_RECIPE["dart"])
    np.save(P("oof_dart.npy"), o); np.save(P("te_dart.npy"), t)

    # 线下核对：三族 rank 等权平均
    R = lambda v: rankdata(v) / len(v)
    oof_blend = np.mean([R(np.load(P(f"oof_{m}.npy"))) for m in
                         ["dart", "abthin_lgbn", "cat"]], axis=0)
    f1, tp = top1_f1(y, oof_blend)
    print(f"\n最终组合 OOF top1%F1={f1:.4f} ({tp}/3000)")
    np.save(P("oof_final.npy"), oof_blend)
    print(f"10 完成，模型已落盘至 {MODEL_DIR}   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
