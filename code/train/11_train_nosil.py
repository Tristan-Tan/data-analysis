# -*- coding: utf-8 -*-
"""11 no-silence 变体两族（**仅 B 榜**）—— 复赛最终配方的第 4、5 个成员

【为什么需要这一步】
B 榜最终提交 0.880637（664/754）的融合配方是 6 个成员：

    dart + abthin_lgbn + cat + nosil_partial×2 + nosil_full×1

前三个由 10_train_models.py 产出，后两个由本脚本产出。
赛时这两族是用一次性探索脚本训的，只保存了 oof/te 的 npy 而没有保存模型，
导致最终成绩无法从落盘模型复现。本脚本把它并入正式链路，
并与三族一样落盘模型。

【两种模式的由来与取舍】（详见说明文档 §一.8）
silence 族占模型 13.8% gain，但其中「绝对尺度量 + 交互项」在 train 与 testB
之间 PSI 高达 0.25~2.6（两批快照相隔约一个月，观察窗口也差 17 天），
跨批次不可比；而 4 个「组内分位」维度修复后 PSI 仅 0.017/0.006，跨批次稳定。

    full     屏蔽全部 22 维 S_*   -> 单族 OOF 2633（弱），但与三族**足够异构**
    partial  只屏蔽 13 维不稳定量 -> 单族 OOF 2755（强，超过 cat 的 2744），
                                     但与三族**过于同质**，单独加入只换 7 张卡

两难的解法不是二选一，而是**同时引入并给 partial 双倍权重**：
partial 保住判别力、full 提供异构性，实测 top754 换动落在有效区间，
线上由 663 提升到 664。

【口径约束】折划分、超参、seed 与 10_train_models.py 的窄树族完全一致，
唯一变量是特征列的剔除范围 —— 这样两族才是「同一模型的不同视角」，
而不是引入新的超参自由度。

输出:
  model/B/nosil_{mode}_s42_f{0..4}.txt      共 10 个模型
  model/B/nosil_{mode}_feature_cols.json    每种模式的列顺序
  data/interim_B/oof_nosil_{mode}.npy       OOF（rank 归一后）
  data/interim_B/te_nosil_{mode}.npy        testB 预测（rank 归一后）
耗时: 约 20 分钟（两族各 5 折窄树）
"""
import json
import os
import sys
import time

import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (P, M, MODEL_DIR, ROUND, FOLD_SEED, N_FOLDS, NOSIL_SEED, TOP_FRAC,
                    LGB_BASE, LGBN_EXTRA, SIL_COLS, NOSIL_DROP_PARTIAL,
                    top1_f1, banner)
from assemble import build_train, build_test_folds

t0 = time.time()
banner("11_train_nosil")
if ROUND != "B":
    raise SystemExit(
        "no-silence 两族只属于 B 榜最终配方，A 榜三族融合不含它们，跳过本步。\n"
        "这不是错误：run_train.sh 会按赛段自动跳过。")

MODES = ("partial", "full")


def dropped_cols(mode, cols):
    """返回该模式下要剔除的列。full = 全部 22 维 S_*；partial = 13 维不稳定量"""
    names = SIL_COLS if mode == "full" else NOSIL_DROP_PARTIAL
    drop = {"S_" + c for c in names}
    missing = drop - set(cols)
    assert not missing, f"特征矩阵里缺少应剔除的列: {sorted(missing)}"
    return drop


def main():
    tr, Xtr, y, cols = build_train()
    te, Xte_list = build_test_folds(N_FOLDS, cols)
    print(f"基础矩阵 train{Xtr.shape} / testB{Xte_list[0].shape}"
          f"   {time.time()-t0:.0f}s")

    # ⚠ 一致性校验：nosil 两族必须与 10_train_models.py 训出的三族**共用同一
    #   套基础列**。若复用了此前训好的三族模型、而本次算出的列有任何出入，
    #   两批模型在 predict 时会对不上，且不会报错——这里提前拦住。
    fc = M("feature_cols.json")
    if os.path.exists(fc):
        with open(fc) as f:
            saved = json.load(f)
        if saved != cols:
            only_saved = [c for c in saved if c not in set(cols)]
            only_now = [c for c in cols if c not in set(saved)]
            raise SystemExit(
                "基础特征列与 model/B/feature_cols.json 不一致，已中止。\n"
                f"  已落盘 {len(saved)} 维 / 本次算出 {len(cols)} 维\n"
                f"  仅在已落盘中: {only_saved[:8]}\n"
                f"  仅在本次中:   {only_now[:8]}\n"
                "顺序不同也算不一致。请确认三族模型与本步跑在同一份中间产物上；"
                "若中间产物已变，请连同 10_train_models.py 一起重跑。")
        print(f"  ✔ 基础列与已落盘的三族模型一致（{len(cols)} 维）")
    else:
        print("  注意：未找到 feature_cols.json，本步无法校验与三族的列一致性；"
              "请确认稍后会运行 10_train_models.py")

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True,
                                 random_state=FOLD_SEED).split(Xtr, y))
    params = dict(LGB_BASE, seed=NOSIL_SEED)
    params.update(LGBN_EXTRA)          # 与 abthin_lgbn 同一套窄树超参

    R = lambda v: rankdata(v) / len(v)                        # noqa: E731
    for mode in MODES:
        drop = dropped_cols(mode, cols)
        use = [c for c in cols if c not in drop]
        kept = [c for c in use if c.startswith("S_")]
        print(f"\n=== nosil_{mode}：{len(cols)} -> {len(use)} 维"
              f"（剔除 {len(cols)-len(use)} 维，保留 silence 维 {len(kept)} 个）===")
        if kept:
            print(f"    保留的 silence 维: {kept}")

        oof = np.zeros(len(Xtr))
        pte = np.zeros(len(Xte_list[0]))
        for k, (ti, vi) in enumerate(folds):
            m = lgb.train(params, lgb.Dataset(Xtr.iloc[ti][use], y[ti]), 3000,
                          valid_sets=[lgb.Dataset(Xtr.iloc[vi][use], y[vi])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            # ⚠ 按 best_iteration 截断后保存，使 predict.py 无需再传迭代数
            m.save_model(M(f"nosil_{mode}_s{NOSIL_SEED}_f{k}.txt"),
                         num_iteration=m.best_iteration)
            oof[vi] = m.predict(Xtr.iloc[vi][use],
                                num_iteration=m.best_iteration)
            pte += m.predict(Xte_list[k][use],
                             num_iteration=m.best_iteration) / N_FOLDS
            print(f"  fold{k} best_iter={m.best_iteration:<5} "
                  f"{time.time()-t0:.0f}s")

        with open(M(f"nosil_{mode}_feature_cols.json"), "w") as f:
            json.dump(use, f, ensure_ascii=False, indent=2)
        np.save(P(f"oof_nosil_{mode}.npy"), R(oof))
        np.save(P(f"te_nosil_{mode}.npy"), R(pte))
        _, tp = top1_f1(y, oof)
        print(f"  nosil_{mode} 单族 OOF {tp}/{int(len(y)*TOP_FRAC)}"
              f"（B 榜真实数据记录值：partial 2755 / full 2633）")

    # ---- 线下核对：最终 6 成员配方的 OOF ----
    fam = ["dart", "abthin_lgbn", "cat"]
    if all(os.path.exists(P(f"oof_{m}.npy")) for m in fam):
        parts3 = [R(np.load(P(f"oof_{m}.npy"))) for m in fam]
        _, tp3 = top1_f1(y, R(np.mean(parts3, axis=0)))
        final = parts3 + [R(np.load(P("oof_nosil_partial.npy")))] * 2 \
            + [R(np.load(P("oof_nosil_full.npy")))]
        _, tp6 = top1_f1(y, R(np.mean(final, axis=0)))
        kk = int(len(y) * TOP_FRAC)
        print(f"\n三族 OOF {tp3}/{kk} -> 最终 6 成员配方 OOF {tp6}/{kk}"
              f"（{tp6-tp3:+d}）")
        print("  ⚠ 线下 OOF 与线上在本赛题已脱钩（B 榜 gap 4.6pt），"
              "该数字仅供核对流程是否跑通，不作为方案优劣依据")
    else:
        print("\n⚠ 未找到三族 OOF，请先运行 10_train_models.py")

    print(f"\n11 完成，模型已落盘至 {MODEL_DIR}   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
