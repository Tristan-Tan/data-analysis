# -*- coding: utf-8 -*-
"""10 训练最终模型并落盘

训练三个基础模型族（两榜共用本脚本）：
    dart          LightGBM DART        seed [42]
    abthin_lgbn   LightGBM 窄树 + AB-thin 7 维   seed [42, 2026]
    cat           CatBoost 对称树      seed [42]

维度随赛段不同：
    A 榜 401 / 408 维（不含 smy_te）   —— 最终成绩 0.917771（692/754）
    B 榜 402 / 409 维（含 smy_te）     —— 三族只是 B 榜最终配方的前三个成员，
                                          还需 11_train_nosil.py 的两个 nosil 族

⚠ 是否含 smy_te 由 config.SPEC["use_smy_te"] 决定，不看文件是否存在。

为什么是这三个而不是「线下最好的那个」：见说明文档 §一.6。
简言之，线下 OOF 分数经过数十轮选择后已带过拟合，2772~2780 的线下差异
在线上只对应随机摆动；而成员多、结构异构的组合，选择性过拟合被平均得最彻底。

⚠ 折划分固定 random_state=42 —— exp5 是在该折下 fold-safe 生成的。
⚠ 第 k 折的模型使用第 k 份 fold-matched 测试编码做预测。

输出（均在赛段专属目录下）:
  model/{A|B}/{family}_s{seed}_f{fold}.txt|cbm   共 20 个模型
  model/{A|B}/feature_cols.json                  基础列顺序
  model/{A|B}/account_balance_thin_feature_cols.json  AB-thin 列顺序
  data/interim_{A|B}/oof_{family}.npy            OOF（供线下核对）
  data/interim_{A|B}/te_{family}.npy             测试侧族预测
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
from config import (P, M, MODEL_DIR, FOLD_SEED, N_FOLDS, FINAL_RECIPE, TOP_FRAC,
                    LGB_BASE, DART_EXTRA, LGBN_EXTRA, CAT_PARAMS,
                    TARGET, ROUND, SPEC, top1_f1, banner)
from assemble import build_train, build_test_folds
from account_balance import (THIN_COLS, append_features,
                             load_or_create_features)

t0 = time.time()
banner("10_train_models")


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
            m.save_model(M(f"{family}_s{sd}_f{k}.txt"),
                         num_iteration=m.best_iteration)
            oof[vi] = m.predict(X.iloc[vi][cols], num_iteration=m.best_iteration)
            pte += m.predict(Xte[k][cols], num_iteration=m.best_iteration) / N_FOLDS
        _, tp = top1_f1(y, oof)
        print(f"  [{family}] seed={sd:<6} OOF {tp}/{int(len(y)*TOP_FRAC)}"
              f"   {time.time()-t0:.0f}s")
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
            m.save_model(M(f"cat_s{sd}_f{k}.cbm"))
            oof[vi] = m.predict_proba(X.iloc[vi][cols])[:, 1]
            pte += m.predict_proba(Xte[k][cols])[:, 1] / N_FOLDS
        _, tp = top1_f1(y, oof)
        print(f"  [cat] seed={sd:<6} OOF {tp}/{int(len(y)*TOP_FRAC)}"
              f"   {time.time()-t0:.0f}s")
        oof_acc += rankdata(oof) / len(oof)
        te_acc += rankdata(pte) / len(pte)
    return oof_acc / len(seeds), te_acc / len(seeds)


def main():
    tr, Xbase, y, base_cols = build_train()
    print(f"基础训练矩阵 {Xbase.shape}   {time.time()-t0:.0f}s")

    # 财富归一自检（口径错了这里立刻会露出来）
    m1 = Xbase.loc[y == 1, "in_vs_aum"].mean()
    m0 = Xbase.loc[y == 0, "in_vs_aum"].mean()
    print(f"自检 in_vs_aum: label1={m1:.2f} label0={m0:.2f}"
          f"  （真实数据应约 178 / 21；合成自检数据不适用）")

    te, Xte_base = build_test_folds(N_FOLDS, base_cols)
    print(f"基础测试矩阵 {Xte_base[0].shape} × {N_FOLDS} 份（fold-matched）")

    ab = load_or_create_features()
    Xthin = append_features(Xbase, tr[["card_no"]], ab)
    Xte_thin = [append_features(x, te[["card_no"]], ab)
                for x in Xte_base]
    thin_cols = base_cols + THIN_COLS
    assert Xthin.columns.tolist() == thin_cols
    print(f"AB-thin 训练/测试矩阵 {Xthin.shape}/{Xte_thin[0].shape}")

    with open(M("feature_cols.json"), "w") as f:
        json.dump(base_cols, f, ensure_ascii=False, indent=2)
    with open(M("account_balance_thin_feature_cols.json"), "w") as f:
        json.dump(thin_cols, f, ensure_ascii=False, indent=2)

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True,
                                 random_state=FOLD_SEED).split(Xbase, y))

    print(f"\n=== abthin_lgbn（{len(base_cols)} + {len(THIN_COLS)} 维 AB-thin）===")
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
    oof_blend = R(np.mean([R(np.load(P(f"oof_{m}.npy"))) for m in
                           ["dart", "abthin_lgbn", "cat"]], axis=0))
    f1, tp = top1_f1(y, oof_blend)
    print(f"\n三族组合 OOF top1%F1={f1:.4f} ({tp}/{int(len(y)*TOP_FRAC)})")
    np.save(P("oof_3fam.npy"), oof_blend)
    print(f"10 完成，模型已落盘至 {MODEL_DIR}   {time.time()-t0:.0f}s")
    if ROUND == "B":
        print("\n⚠ B 榜最终配方还需两个 no-silence 族，"
              "请接着运行 code/train/11_train_nosil.py")


if __name__ == "__main__":
    main()
