# -*- coding: utf-8 -*-
"""实验 1：联合事件 / 账户内序列特征的增量验证

【唯一变量】特征集。基线与实验用**同一模型类型（lgbn 窄树）、同一折划分
（FOLD_SEED=42，与 exp5 的 fold-safe 生成口径一致）、同一超参、同一 seed**，
差别只有是否加入 f01 的 J_* 列，因此增量可直接归因。

  baseline : 现有 402 维
  exp      : 402 维 + f01 联合事件/序列特征

【输出】按验收要求逐项打印：
  1 全局 OOF Top1% 命中 + 各折结果
  2 相对基线的换入/换出（m 指每侧数量）
  3 OOF 换入正例、换出正例、净收益
  4 testB Top754 重叠、换入卡的基线原排名范围
  5 基线边界区域条件区分能力（子集内 AUC）+ 候选池召回上限
  6 新入选卡由哪些新增信号支持
  7 数据覆盖、缺失率、特征/模型版本、缓存来源

所有产物写入 data/interim_labB 与 prediction_result/labB，不触碰正式目录。
**不自动提交。**

前置: 已跑过 10_train_models.py testb（需要 te_*.npy 供融合候选）
      已跑过 code/labB/f01_joint_events.py testb
用法: PYTHONPATH=code python code/labB/run_exp1.py testb
耗时: 约 15~20 分钟（两个 lgbn 族，各 5 折）
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
                        coverage_report, data_manifest, load_cached)
import assemble

t0 = time.time()
TARGET = sys.argv[1] if len(sys.argv) > 1 else "testb"
SEED = FOLD_SEED
FAMILIES = ["dart", "abthin_lgbn", "cat"]

# ---------------- 数据 ----------------
tr, Xtr, y, cols = assemble.build_train()
te, Xte_list = assemble.build_test_folds(N_FOLDS, cols)
print(f"基线矩阵 train{Xtr.shape} / {TARGET}{Xte_list[0].shape}"
      f"   {time.time()-t0:.0f}s")

jf = pd.read_parquet(LP("f01_joint.parquet"))
jf["card_no"] = jf["card_no"].astype(str)
jcols = [c for c in jf.columns if c.startswith("J_")]
with open(LP("f01_joint.parquet.manifest.json"), encoding="utf-8") as f:
    j_mani = json.load(f)
print(f"f01 特征 {len(jcols)} 列，来源 manifest: "
      f"feature_version={j_mani['feature_version']}, target={j_mani['target']}")
assert j_mani["target"] == TARGET, "f01 的 target 与本次实验不一致，请重算 f01"


def attach(X, keys, frame, fcols):
    a = keys[["card_no"]].copy()
    a["card_no"] = a["card_no"].astype(str)
    a = a.merge(frame, on="card_no", how="left", validate="one_to_one")
    v = a[fcols].astype(np.float32)
    v.index = X.index
    out = pd.concat([X, v], axis=1)
    assert len(out) == len(X), "拼接后行数变化"
    return out


Xtr_e = attach(Xtr, tr, jf, jcols)
Xte_e = [attach(x, te, jf, jcols) for x in Xte_list]
exp_cols = cols + jcols
print(f"实验矩阵 train{Xtr_e.shape} / {TARGET}{Xte_e[0].shape}")

# 7) 覆盖与缺失
coverage_report(Xtr_e, jcols, "train 侧 J_*")
coverage_report(Xte_e[0], jcols, f"{TARGET} 侧 J_*")

folds = list(StratifiedKFold(N_FOLDS, shuffle=True,
                             random_state=FOLD_SEED).split(Xtr, y))
params = dict(LGB_BASE, seed=SEED)
params.update(LGBN_EXTRA)


def train(X, Xte, use_cols, tag):
    oof = np.zeros(len(X))
    pte = np.zeros(len(Xte[0]))
    imp = np.zeros(len(use_cols))
    fold_hits = []
    for k, (ti, vi) in enumerate(folds):
        m = lgb.train(params, lgb.Dataset(X.iloc[ti][use_cols], y[ti]), 3000,
                      valid_sets=[lgb.Dataset(X.iloc[vi][use_cols], y[vi])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[vi] = m.predict(X.iloc[vi][use_cols], num_iteration=m.best_iteration)
        pte += m.predict(Xte[k][use_cols],
                         num_iteration=m.best_iteration) / N_FOLDS
        imp += m.feature_importance("gain") / N_FOLDS
        kk = int(len(vi) * TOP_FRAC)
        fold_hits.append(int(y[vi][np.argsort(-oof[vi])[:kk]].sum()))
        print(f"  [{tag}] fold{k} best_iter={m.best_iteration} "
              f"折内 top1% 命中 {fold_hits[-1]}/{kk}   {time.time()-t0:.0f}s")
    return oof, pte, imp, fold_hits


oof_b, te_b, imp_b, fh_b = train(Xtr, Xte_list, cols, "baseline")
oof_e, te_e, imp_e, fh_e = train(Xtr_e, Xte_e, exp_cols, "exp")

# ---------------- 1) OOF ----------------
_, tp_b = top1_f1(y, oof_b)
_, tp_e = top1_f1(y, oof_e)
print("\n" + "=" * 74)
print(f"【1】全局 OOF Top1% 命中：baseline {tp_b}/3000 -> exp {tp_e}/3000"
      f"（{tp_e - tp_b:+d}）")
print(f"     各折：baseline {fh_b}  合计 {sum(fh_b)}")
print(f"           exp      {fh_e}  合计 {sum(fh_e)}")
print(f"     逐折差: {[e - b for e, b in zip(fh_e, fh_b)]}")

# ---------------- 2/3) OOF 换卡 ----------------
print("\n【2/3】OOF 侧换卡与净收益")
res_oof, sin, sout = swap_report(oof_b, oof_e, y=y, label="OOF")

# ---------------- 5) 边界区分能力 + 候选池召回 ----------------
print("\n【5】边界区分能力与候选池召回")
boundary_auc(y, oof_b, oof_e, 500, 1500)
boundary_auc(y, oof_b, oof_e, 1500, 6000)
br = rankdata(-oof_b, method="ordinal")
for frac in (0.02, 0.03):
    kk = int(len(y) * frac)
    rec = y[br <= kk].sum() / y.sum()
    print(f"  基线 Top{frac:.0%}（{kk} 张）对训练正例的召回上限: {rec:.2%}")

# ---------------- 6) 新入选卡的信号归因 ----------------
print("\n【6】新入选卡由哪些新增信号支持")
imp_s = pd.Series(imp_e, index=exp_cols).sort_values(ascending=False)
j_rank = {c: int(np.where(imp_s.index == c)[0][0]) + 1 for c in jcols}
top_j = sorted(jcols, key=lambda c: -imp_s[c])[:8]
print(f"  J_* 在 exp 模型 gain 排名最高的 8 个"
      f"（全 {len(exp_cols)} 维中）：")
for c in top_j:
    share = imp_s[c] / imp_s.sum()
    print(f"    {c:<28} rank={j_rank[c]:<5} gain占比={share:.3%}")
print(f"  J_* 合计 gain 占比: {imp_s[jcols].sum() / imp_s.sum():.3%}")
if sin:
    print("  换入卡 vs 全体 在上述特征上的中位数：")
    for c in top_j[:5]:
        v = Xtr_e[c]
        print(f"    {c:<28} 换入 {np.nanmedian(v.iloc[sin]):.4f}  "
              f"全体 {np.nanmedian(v):.4f}")

# ---------------- 4) testB 侧 ----------------
print("\n【4】" + TARGET + " 侧换卡（单模型对照，非最终提交口径）")
swap_report(te_b, te_e, y=None, label=f"{TARGET} 单模型")

np.save(LP(f"oof_exp1_{TARGET}.npy"), R(oof_e))
np.save(LP(f"te_exp1_{TARGET}.npy"), R(te_e))
np.save(LP(f"oof_exp1base_{TARGET}.npy"), R(oof_b))
np.save(LP(f"te_exp1base_{TARGET}.npy"), R(te_b))

# ---------------- 融合候选（族内平均后按固定族权重）----------------
print("\n【融合候选】现有三族 + exp 作为第四族（族权重等权，不按实例数）")
ok = all(os.path.exists(P(f"te_{m}.npy")) for m in FAMILIES)
if not ok:
    print("  缺少正式 te_*.npy，跳过融合候选")
else:
    fam_te = {m: np.load(P(f"te_{m}.npy")) for m in FAMILIES}
    fam_oof = {m: np.load(P(f"oof_{m}.npy")) for m in FAMILIES}
    base3_te = R(np.mean([R(fam_te[m]) for m in FAMILIES], axis=0))
    base3_oof = R(np.mean([R(fam_oof[m]) for m in FAMILIES], axis=0))
    _, tp3 = top1_f1(y, base3_oof)
    new4_oof = R(np.mean([R(fam_oof[m]) for m in FAMILIES] + [R(oof_e)], axis=0))
    new4_te = R(np.mean([R(fam_te[m]) for m in FAMILIES] + [R(te_e)], axis=0))
    _, tp4 = top1_f1(y, new4_oof)
    print(f"  三族 OOF {tp3}/3000 -> 四族(+exp) {tp4}/3000（{tp4 - tp3:+d}）")
    swap_report(base3_oof, new4_oof, y=y, label="融合 OOF")

    sil = pd.read_parquet(P("silence_feat.parquet"),
                          columns=["card_no", "sil_days"])
    sil["card_no"] = sil["card_no"].astype(str)
    hard = te.merge(sil, on="card_no", how="left")["sil_days"].values >= 32
    s = new4_te.copy()
    b = base3_te.copy()
    if hard.any():
        s[hard] = 1.0
        b[hard] = 1.0
    swap_report(b, s, y=None, label=f"融合 {TARGET}")
    sub = te[["card_no"]].copy()
    sub["score"] = s
    out = os.path.join(LAB_PRED, f"pred_{TARGET}_exp1_4fam.csv")
    sub.to_csv(out, index=False)
    print(f"  候选文件（未提交）: {out}")

print(f"\n全部完成 {time.time()-t0:.0f}s")
