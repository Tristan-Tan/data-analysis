# -*- coding: utf-8 -*-
"""实验 2：账户内资金路径 + 全窗口异常片段特征的增量验证

【唯一变量】特征集。基线与实验用同一模型类型（lgbn 窄树）、同一折划分
（FOLD_SEED=42）、同一超参、同一 seed，差别只有是否加入 f02 的 F_* 列。

  baseline : 现有 402 维
  exp      : 402 维 + f02 资金路径/异常片段特征

与实验 1 的区别：f01 是**事件类型的组合**（方向×摘要×渠道、相邻事件），
f02 是**金额层面的流向**（FIFO 匹配出的覆盖时间、覆盖比例、目的地）。
两者信息源不同，实验 1 的负结果不预判实验 2。

【诊断口径与实验 1 保持一致，便于横向对比】
边界区间新增 [2500,6000)：实验 1 显示 [500,1500) 子集正例 994/1000、
仅 6 个负例，AUC 不可靠；真正的战场是 top3000 前后那 202 个正例。

所有产物写入 data/interim_labB 与 prediction_result/labB。**不自动提交。**

前置: 10_train_models.py testb 已跑过；code/labB/f02_fund_path.py testb 已跑过
用法: PYTHONPATH=code python code/labB/run_exp2.py testb
耗时: 约 15~20 分钟
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (P, FOLD_SEED, N_FOLDS, LGB_BASE, LGBN_EXTRA, TOP_FRAC,
                    top1_f1)
from lab_common import (LP, LAB_PRED, R, swap_report, boundary_auc,
                        coverage_report)
import assemble

t0 = time.time()
TARGET = sys.argv[1] if len(sys.argv) > 1 else "testb"
FEAT_FILE, PREFIX, TAG = "f02_fund.parquet", "F_", "exp2"
FAMILIES = ["dart", "abthin_lgbn", "cat"]

tr, Xtr, y, cols = assemble.build_train()
te, Xte_list = assemble.build_test_folds(TARGET, N_FOLDS, cols)
print(f"基线矩阵 train{Xtr.shape} / {TARGET}{Xte_list[0].shape}"
      f"   {time.time()-t0:.0f}s")

ff = pd.read_parquet(LP(FEAT_FILE))
ff["card_no"] = ff["card_no"].astype(str)
fcols = [c for c in ff.columns if c.startswith(PREFIX)]
with open(LP(FEAT_FILE + ".manifest.json"), encoding="utf-8") as f:
    mani = json.load(f)
print(f"{FEAT_FILE} 特征 {len(fcols)} 列，manifest: "
      f"feature_version={mani['feature_version']}, target={mani['target']}")
assert mani["target"] == TARGET, "特征文件的 target 与本次实验不一致"


def attach(X, keys, frame, use):
    a = keys[["card_no"]].copy()
    a["card_no"] = a["card_no"].astype(str)
    a = a.merge(frame, on="card_no", how="left", validate="one_to_one")
    v = a[use].astype(np.float32)
    v.index = X.index
    out = pd.concat([X, v], axis=1)
    assert len(out) == len(X), "拼接后行数变化"
    return out


Xtr_e = attach(Xtr, tr, ff, fcols)
Xte_e = [attach(x, te, ff, fcols) for x in Xte_list]
exp_cols = cols + fcols
print(f"实验矩阵 train{Xtr_e.shape} / {TARGET}{Xte_e[0].shape}")

coverage_report(Xtr_e, fcols, f"train 侧 {PREFIX}*")
coverage_report(Xte_e[0], fcols, f"{TARGET} 侧 {PREFIX}*")

folds = list(StratifiedKFold(N_FOLDS, shuffle=True,
                             random_state=FOLD_SEED).split(Xtr, y))
params = dict(LGB_BASE, seed=FOLD_SEED)
params.update(LGBN_EXTRA)


def train(X, Xte, use_cols, tag):
    oof = np.zeros(len(X))
    pte = np.zeros(len(Xte[0]))
    imp = np.zeros(len(use_cols))
    hits = []
    for k, (ti, vi) in enumerate(folds):
        m = lgb.train(params, lgb.Dataset(X.iloc[ti][use_cols], y[ti]), 3000,
                      valid_sets=[lgb.Dataset(X.iloc[vi][use_cols], y[vi])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[vi] = m.predict(X.iloc[vi][use_cols], num_iteration=m.best_iteration)
        pte += m.predict(Xte[k][use_cols],
                         num_iteration=m.best_iteration) / N_FOLDS
        imp += m.feature_importance("gain") / N_FOLDS
        kk = int(len(vi) * TOP_FRAC)
        hits.append(int(y[vi][np.argsort(-oof[vi])[:kk]].sum()))
        print(f"  [{tag}] fold{k} best_iter={m.best_iteration} "
              f"折内 top1% 命中 {hits[-1]}/{kk}   {time.time()-t0:.0f}s")
    return oof, pte, imp, hits


oof_b, te_b, _, fh_b = train(Xtr, Xte_list, cols, "baseline")
oof_e, te_e, imp_e, fh_e = train(Xtr_e, Xte_e, exp_cols, TAG)

_, tp_b = top1_f1(y, oof_b)
_, tp_e = top1_f1(y, oof_e)
print("\n" + "=" * 74)
print(f"【1】全局 OOF Top1% 命中：baseline {tp_b}/3000 -> {TAG} {tp_e}/3000"
      f"（{tp_e - tp_b:+d}）")
print(f"     各折：baseline {fh_b}  合计 {sum(fh_b)}")
print(f"           {TAG}      {fh_e}  合计 {sum(fh_e)}")
print(f"     逐折差: {[e - b for e, b in zip(fh_e, fh_b)]}")

print("\n【2/3】OOF 侧换卡与净收益")
_, sin, sout = swap_report(oof_b, oof_e, y=y, label="OOF")

print("\n【5】边界区分能力与候选池召回")
boundary_auc(y, oof_b, oof_e, 1500, 6000)
boundary_auc(y, oof_b, oof_e, 2500, 6000)
br = rankdata(-oof_b, method="ordinal")
for frac in (0.02, 0.03):
    kk = int(len(y) * frac)
    print(f"  基线 Top{frac:.0%}（{kk} 张）对训练正例的召回上限: "
          f"{y[br <= kk].sum() / y.sum():.2%}")

print(f"\n【6】新入选卡由哪些新增信号支持")
imp_s = pd.Series(imp_e, index=exp_cols).sort_values(ascending=False)
top_f = sorted(fcols, key=lambda c: -imp_s[c])[:8]
for c in top_f:
    rank = int(np.where(imp_s.index == c)[0][0]) + 1
    print(f"    {c:<26} rank={rank:<5} gain占比={imp_s[c] / imp_s.sum():.3%}")
print(f"  {PREFIX}* 合计 gain 占比: {imp_s[fcols].sum() / imp_s.sum():.3%}")
if sin:
    print("  换入卡 vs 全体（用均值，稀疏列中位数无区分度）：")
    for c in top_f[:5]:
        v = Xtr_e[c].values
        print(f"    {c:<26} 换入 {np.nanmean(v[sin]):.4f}  "
              f"全体 {np.nanmean(v):.4f}")

print(f"\n【4】{TARGET} 侧换卡（单模型对照，非最终提交口径）")
swap_report(te_b, te_e, y=None, label=f"{TARGET} 单模型")

np.save(LP(f"oof_{TAG}_{TARGET}.npy"), R(oof_e))
np.save(LP(f"te_{TAG}_{TARGET}.npy"), R(te_e))

print("\n【融合候选】三族 + exp 作为第四族（族内平均后固定族权重等权）")
if not all(os.path.exists(P(f"te_{m}.npy")) for m in FAMILIES):
    print("  缺少正式 te_*.npy，跳过")
else:
    fte = {m: np.load(P(f"te_{m}.npy")) for m in FAMILIES}
    foof = {m: np.load(P(f"oof_{m}.npy")) for m in FAMILIES}
    b_oof = R(np.mean([R(foof[m]) for m in FAMILIES], axis=0))
    b_te = R(np.mean([R(fte[m]) for m in FAMILIES], axis=0))
    n_oof = R(np.mean([R(foof[m]) for m in FAMILIES] + [R(oof_e)], axis=0))
    n_te = R(np.mean([R(fte[m]) for m in FAMILIES] + [R(te_e)], axis=0))
    _, t3 = top1_f1(y, b_oof)
    _, t4 = top1_f1(y, n_oof)
    print(f"  三族 OOF {t3}/3000 -> 四族 {t4}/3000（{t4 - t3:+d}）")
    swap_report(b_oof, n_oof, y=y, label="融合 OOF")

    sil = pd.read_parquet(P("silence_feat.parquet"),
                          columns=["card_no", "sil_days"])
    sil["card_no"] = sil["card_no"].astype(str)
    hard = te.merge(sil, on="card_no", how="left")["sil_days"].values >= 32
    s, b = n_te.copy(), b_te.copy()
    if hard.any():
        s[hard] = 1.0
        b[hard] = 1.0
    swap_report(b, s, y=None, label=f"融合 {TARGET}")
    sub = te[["card_no"]].copy()
    sub["score"] = s
    out = os.path.join(LAB_PRED, f"pred_{TARGET}_{TAG}_4fam.csv")
    sub.to_csv(out, index=False)
    print(f"  候选文件（未提交）: {out}")

print(f"\n全部完成 {time.time()-t0:.0f}s")
