# -*- coding: utf-8 -*-
"""【已被正式链路取代，保留作过程记录】

本脚本是赛中训练 no-silence 两个变体的探索版本，**只保存 oof/te 的 npy，
不保存模型** —— 这正是复赛最终成绩一度无法从落盘模型复现的原因。

正式复现请用: code/train/11_train_nosil.py（同一套超参与折划分，
但会把 10 个模型落盘到 model/B/，供 predict.py 直接加载）。

本文件仅用于追溯当时的判断过程，不参与复现链路。
运行前需 export FRAUD_ROUND=B。

no-silence 变体：训练一个屏蔽 silence 族的窄树，作为第四个融合成员

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

【两种模式】
  full    屏蔽全部 22 维 S_*。实测 OOF 2633/3000（比最弱的 cat 还低 111），
          四族融合 OOF -18，属于过度切除：silence 族里的 4 个组内分位特征
          修复后 PSI 仅 0.017/0.006（跨批次完全稳定）且 gain 合计 9%+，
          是全族最强且唯一稳定的部分，不该被一起扔掉。
  partial 只屏蔽 13 维「绝对尺度量 + 交互项」（PSI 0.25~2.6 的全部在内），
          保留 4 个组内分位 + 5 个时间标志。既去掉不稳定部分又保住强特征，
          OOF 损失应远小于 full。**推荐用这个。**

前置: 已完成一轮 10_train_models.py testb（需要 te_*.npy / oof_*.npy）
用法: PYTHONPATH=code python code/explore/no_silence_variant.py partial
      PYTHONPATH=code python code/explore/no_silence_variant.py full
耗时: 约 10 分钟（只训一个窄树族）
输出: prediction_result/prediction_resultB_nosil4fam_{mode}.csv（不覆盖现有文件）
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
                    SIL_COLS, top1_f1)
import assemble

t0 = time.time()
FAMILIES = ["dart", "abthin_lgbn", "cat"]
MODE = sys.argv[1] if len(sys.argv) > 1 else "partial"
assert MODE in ("full", "partial"), "模式只能是 full 或 partial"

# 绝对尺度量与交互项：PSI 0.25~2.6，跨批次不可比
DROP_PARTIAL = ["sil_days", "sil_hours", "first_sil_days", "sil_div_span",
                "ntxn_x_sil", "ntxn_div_sil", "insum_x_sil", "bal_x_sil",
                "t1share_x_sil", "t3share_x_sil", "netflow_x_sil",
                "retention_x_sil", "txnrate_x_sil"]


def R(v):
    return rankdata(v) / len(v)


# ---------------- 特征矩阵：按模式剔除 silence 相关列 ----------------
tr, Xtr, y, cols = assemble.build_train()
drop = {"S_" + c for c in (SIL_COLS if MODE == "full" else DROP_PARTIAL)}
nosil_cols = [c for c in cols if c not in drop]
kept_sil = [c for c in cols if c.startswith("S_") and c not in drop]
print(f"[模式 {MODE}] train 矩阵 {Xtr.shape} -> {len(nosil_cols)} 维"
      f"（剔除 {len(cols) - len(nosil_cols)} 维）")
print(f"  保留的 silence 维度 {len(kept_sil)} 个: {kept_sil}")
print(f"  {time.time()-t0:.0f}s")

te, Xte_list = assemble.build_test_folds(N_FOLDS, cols)
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
np.save(P(f"oof_nosil_{MODE}.npy"), R(oof))
np.save(P(f"te_nosil_{MODE}.npy"), R(pte))

# ---------------- OOF 对比 ----------------
print("\n" + "=" * 70)
print("【OOF 对比】")
oof_fam = {}
for m in FAMILIES:
    p = P(f"oof_{m}.npy")
    if os.path.exists(p):
        oof_fam[m] = np.load(p)
        _, tp = top1_f1(y, oof_fam[m])
        print(f"  {m:<16} OOF {tp}/3000")
print(f"  {'nosil_' + MODE:<16} OOF {tp_nosil}/3000   ← 判断点")

if len(oof_fam) == 3:
    b3 = R(np.mean([R(oof_fam[m]) for m in FAMILIES], axis=0))
    _, tp3 = top1_f1(y, b3)
    print(f"\n  三族融合 OOF {tp3}/3000")
    for mode in ("full", "partial"):
        pm = P(f"oof_nosil_{mode}.npy")
        if not os.path.exists(pm):
            continue
        om = np.load(pm)
        b4 = R(np.mean([R(oof_fam[m]) for m in FAMILIES] + [R(om)], axis=0))
        _, tp4 = top1_f1(y, b4)
        print(f"  四族融合(+{mode:<8}) OOF {tp4}/3000   （差 {tp4 - tp3:+d}）")

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
    out = os.path.join(PRED_DIR, f"prediction_resultB_nosil4fam_{MODE}.csv")
    sub.to_csv(out, index=False)
    print(f"已写出 {out}   {sub.shape}")

    # 与现有三族提交的差异：top754 集合换了多少张
    s3 = R(np.mean([R(te_fam[m]) for m in FAMILIES], axis=0))
    if n_hard:
        s3[sv >= 32] = 1.0
    k = int(len(score) * 0.01)
    top3 = set(np.argsort(-s3)[:k])
    print(f"\n【与已提交的三族版(662/754)相比】")
    print(f"  四族(+{MODE})：top{k} 里有 {k - len(set(np.argsort(-score)[:k]) & top3)}"
          f" 张卡不同（重叠 {len(set(np.argsort(-score)[:k]) & top3)}/{k}）")

    # 若两种模式都已跑过，一并对比，便于二选一
    other = "full" if MODE == "partial" else "partial"
    po = P(f"te_nosil_{other}.npy")
    if os.path.exists(po):
        so = R(np.mean([R(te_fam[m]) for m in FAMILIES] + [R(np.load(po))],
                       axis=0))
        if n_hard:
            so[sv >= 32] = 1.0
        ka = set(np.argsort(-score)[:k])
        ko = set(np.argsort(-so)[:k])
        print(f"  四族(+{other})：top{k} 里有 {k - len(ko & top3)} 张卡不同"
              f"（重叠 {len(ko & top3)}/{k}）")
        print(f"  两个变体之间：top{k} 互不相同 {k - len(ka & ko)} 张")
    print("\n  换动 <20 张 -> 方向对也动不了分；20~150 合适；>150 激进但信息量大")

print(f"\n全部完成 {time.time()-t0:.0f}s")
