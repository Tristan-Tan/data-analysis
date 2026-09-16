# -*- coding: utf-8 -*-
"""no-silence 变体：训练一个屏蔽 silence 族的窄树，作为第四个融合成员

【为什么做这个】
B 榜两次提交 663/754 -> 662/754，组内分位修复在线上零收益。原因已想清楚：
评分只看 testB **内部**的相对排序，而那个修复本质是保序的尺度变换（修复前
testB 分位被压在 0.41 均值，修复后铺满到 0.50，但同箱内先后顺序不变），
保序变换不改变 top754 是哪些卡。

推论：**高 PSI ≠ 掉分**。silence 交互项族那些 PSI 0.25~2.6 的偏移大概率
同性质，再花 2.5 小时去改很可能是第二次撞同一堵墙。

真正的问题猜想：testB 观察窗口比 train 短 17 天（42 天 vs 59 天），而
silence 族的逻辑是"欺诈卡被止付后沉默天数显著更高"（train label1 均值
16.6 天 vs label0 7.9 天）——窗口短意味着很多 testB 欺诈卡还没积累够沉默
证据就被截断，该族在 testB 上判别力天然打折，而模型有 13.8% gain 押在上面。

【做法】屏蔽全部 S_* 共 22 维，训练 lgbn 窄树 5 折，作为第四个成员加入
rank 等权融合。dart / cat / abthin_lgbn 完全复用现有 te_*.npy，不重训。

【判断点】训完立刻能看到 no-silence 单族 OOF：
  · 只掉一点（~2700 上下）-> silence 并非不可替代，融合风险小，值得提交
  · 掉很多（<2500）       -> 它确实是主心骨，融进去会拖累，放弃该方向

前置: 已完成一轮 10_train_models.py testb（需要 te_*.npy / oof_*.npy）
用法: PYTHONPATH=code python code/explore/no_silence_variant.py
耗时: 约 10 分钟（只训一个窄树族）
输出: prediction_result/prediction_resultB_nosil4fam.csv（不覆盖现有文件）
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (P, PRED_DIR, FOLD_SEED, N_FOLDS, LGB_BASE, LGBN_EXTRA,
                    top1_f1)
import assemble

t0 = time.time()
FAMILIES = ["dart", "abthin_lgbn", "cat"]


def R(v):
    return rankdata(v) / len(v)


# ---------------- 特征矩阵：剔除 silence 族 ----------------
tr, Xtr, y, cols = assemble.build_train()
nosil_cols = [c for c in cols if not c.startswith("S_")]
print(f"train 矩阵 {Xtr.shape} -> 屏蔽 silence 后 {len(nosil_cols)} 维"
      f"（剔除 {len(cols) - len(nosil_cols)} 维）   {time.time()-t0:.0f}s")

te, Xte_list = assemble.build_test_folds("testb", N_FOLDS, cols)
print(f"testB 矩阵 {Xte_list[0].shape} × {N_FOLDS} 份   {time.time()-t0:.0f}s")

# ---------------- 训练 no-silence 窄树 ----------------
folds = list(StratifiedKFold(N_FOLDS, shuffle=True,
                             random_state=FOLD_SEED).split(Xtr, y))
params = dict(LGB_BASE, seed=FOLD_SEED)
params.update(LGBN_EXTRA)

oof = np.zeros(len(Xtr))
pte = np.zeros(len(Xte_list[0]))
for k, (ti, vi) in enumerate(folds):
    m = lgb.train(params,
                  lgb.Dataset(Xtr.iloc[ti][nosil_cols], y[ti]), 3000,
                  valid_sets=[lgb.Dataset(Xtr.iloc[vi][nosil_cols], y[vi])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[vi] = m.predict(Xtr.iloc[vi][nosil_cols],
                        num_iteration=m.best_iteration)
    pte += m.predict(Xte_list[k][nosil_cols],
                     num_iteration=m.best_iteration) / N_FOLDS
    print(f"  fold{k} best_iter={m.best_iteration}   {time.time()-t0:.0f}s")

f1_nosil, tp_nosil = top1_f1(y, oof)
np.save(P("oof_nosil.npy"), R(oof))
np.save(P("te_nosil.npy"), R(pte))

# ---------------- OOF 对比 ----------------
print("\n" + "=" * 70)
print("【OOF 对比】")
oof_fam = {}
for m in FAMILIES:
    p = P(f"oof_{m}.npy")
    if os.path.exists(p):
        oof_fam[m] = np.load(p)
        _, tp = top1_f1(y, oof_fam[m])
        print(f"  {m:<14} OOF {tp}/3000")
print(f"  {'nosil(新)':<14} OOF {tp_nosil}/3000   ← 判断点")

if len(oof_fam) == 3:
    b3 = R(np.mean([R(oof_fam[m]) for m in FAMILIES], axis=0))
    _, tp3 = top1_f1(y, b3)
    b4 = R(np.mean([R(oof_fam[m]) for m in FAMILIES] + [R(oof)], axis=0))
    _, tp4 = top1_f1(y, b4)
    print(f"\n  三族融合 OOF {tp3}/3000")
    print(f"  四族融合 OOF {tp4}/3000   （差 {tp4 - tp3:+d}）")

# ---------------- 生成四族融合提交文件 ----------------
te_fam = {}
missing = [m for m in FAMILIES if not os.path.exists(P(f"te_{m}.npy"))]
if missing:
    print(f"\n⚠ 缺少 {missing} 的 te_*.npy，无法生成融合文件，"
          "请确认已跑过 10_train_models.py testb")
else:
    for m in FAMILIES:
        te_fam[m] = np.load(P(f"te_{m}.npy"))
    score = R(np.mean([R(te_fam[m]) for m in FAMILIES] + [R(pte)], axis=0))

    # 硬规则，与 predict.py 完全一致
    sil = pd.read_parquet(P("silence_feat.parquet"),
                          columns=["card_no", "sil_days"])
    sil["card_no"] = sil["card_no"].astype(str)
    sv = te.merge(sil, on="card_no", how="left")["sil_days"].values
    n_hard = int((sv >= 32).sum())
    if n_hard:
        score[sv >= 32] = 1.0
    print(f"\n硬规则命中 {n_hard} 张（sil_days >= 32）")

    sub = te[["card_no"]].copy()
    sub["score"] = score
    assert len(sub) == len(te)
    assert (sub["card_no"].values == te["card_no"].values).all()
    out = os.path.join(PRED_DIR, "prediction_resultB_nosil4fam.csv")
    sub.to_csv(out, index=False)
    print(f"已写出 {out}   {sub.shape}")

    # 与现有三族提交的差异：top754 集合换了多少张
    s3 = R(np.mean([R(te_fam[m]) for m in FAMILIES], axis=0))
    if n_hard:
        s3[sv >= 32] = 1.0
    k = int(len(score) * 0.01)
    a = set(np.argsort(-score)[:k])
    b = set(np.argsort(-s3)[:k])
    print(f"\n四族 vs 三族：top{k} 里有 {k - len(a & b)} 张卡不同"
          f"（重叠 {len(a & b)}/{k} = {len(a & b) / k:.2%}）")
    print("  换动张数太少(<20) -> 即使方向对，分数也动不了多少；"
          "太多(>150) -> 风险较大，但信息量也大")

print(f"\n全部完成 {time.time()-t0:.0f}s")
