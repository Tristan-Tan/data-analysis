# -*- coding: utf-8 -*-
"""多 seed 降方差：唯一不依赖任何假设的方向

【依据】
反复印证的事实：top754 边界处分数跨度仅占全量极差的 0.027%。边界卡挤得
极密，谁进 top754 对模型随机性（bagging_fraction / feature_fraction 的抽样）
极度敏感——三次提交在 662/663 之间横跳正是这个原因。

现有配方只有 4 个模型实例（dart[42]、lgbn[42,2026]、cat[42]）。增加 seed 做
rank 平均能让边界排序更稳定。A 榜的 FINAL_RECIPE 给 lgbn 配 2 个 seed，本身
就是认可这个做法。

【天花板要说清楚】
多 seed 提升的是**稳定性**不是**能力**：期望上它让结果收敛到真实期望值。
若当前 663 恰好低于期望，能捞回几张；若高于期望，反而会掉。不是稳赚。

【实测单 seed 耗时】（取自上一轮 10_train_models.py 的日志）
    abthin_lgbn ~5 分钟   cat ~6.4 分钟   dart ~62 分钟
默认 4 个 lgbn + 3 个 cat ≈ 40 分钟；传第三个参数可追加 dart seed。
每个 seed 训完立即落盘，中途停止也能用已完成的部分；重跑会跳过已存在的。

【融合口径】
现有 te_abthin_lgbn.npy = mean(R(s42), R(s2026)) 即 2 个实例的平均，
te_cat.npy = R(s42) 即 1 个，te_dart.npy = R(s42) 即 1 个。追加 n 个新 seed
后按实例数加权合并：new = (n_old * 旧 + sum(R(新))) / (n_old + n)。
再三族 rank 等权平均，硬规则与 predict.py 一致。

用法:
  PYTHONPATH=code python code/explore/more_seeds.py
  PYTHONPATH=code python code/explore/more_seeds.py 7,2025 2026,7 7
      三个位置参数依次为 lgbn / cat / dart 的新 seed 列表（逗号分隔）
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (P, PRED_DIR, FOLD_SEED, N_FOLDS, LGB_BASE, LGBN_EXTRA,
                    DART_EXTRA, CAT_PARAMS, TOP_FRAC, top1_f1)
from account_balance import THIN_COLS, append_features, load_or_create_features
import assemble

t0 = time.time()


def arg(i, default):
    if len(sys.argv) > i and sys.argv[i].strip():
        return [int(s) for s in sys.argv[i].split(",")]
    return default


NEW = {"lgbn": arg(1, [7, 2025, 1234, 88]),
       "cat": arg(2, [2026, 7, 99]),
       "dart": arg(3, [])}
# 每个族已有的实例数与对应的现成 npy 名
OLD = {"lgbn": (2, "abthin_lgbn"), "cat": (1, "cat"), "dart": (1, "dart")}


def R(v):
    return rankdata(v) / len(v)


# ---------------- 矩阵 ----------------
tr, Xbase, y, base_cols = assemble.build_train()
te, Xte_base = assemble.build_test_folds("testb", N_FOLDS, base_cols)
ab = load_or_create_features()
Xthin = append_features(Xbase, tr[["card_no"]], ab)
Xte_thin = [append_features(x, te[["card_no"]], ab) for x in Xte_base]
thin_cols = base_cols + THIN_COLS
folds = list(StratifiedKFold(N_FOLDS, shuffle=True,
                             random_state=FOLD_SEED).split(Xbase, y))
print(f"矩阵就绪 base{Xbase.shape} thin{Xthin.shape}   {time.time()-t0:.0f}s")
print(f"新增 seed: " + "  ".join(f"{k}={v}" for k, v in NEW.items() if v))


def run_lgb(sd, extra, X, Xte, cols):
    oof = np.zeros(len(X))
    pte = np.zeros(len(Xte[0]))
    p = dict(LGB_BASE, seed=sd)
    p.update(extra)
    for k, (ti, vi) in enumerate(folds):
        m = lgb.train(p, lgb.Dataset(X.iloc[ti][cols], y[ti]), 3000,
                      valid_sets=[lgb.Dataset(X.iloc[vi][cols], y[vi])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[vi] = m.predict(X.iloc[vi][cols], num_iteration=m.best_iteration)
        pte += m.predict(Xte[k][cols], num_iteration=m.best_iteration) / N_FOLDS
    return oof, pte


def run_cat(sd):
    oof = np.zeros(len(Xbase))
    pte = np.zeros(len(Xte_base[0]))
    for k, (ti, vi) in enumerate(folds):
        m = CatBoostClassifier(random_seed=sd, **CAT_PARAMS)
        m.fit(Xbase.iloc[ti][base_cols], y[ti],
              eval_set=(Xbase.iloc[vi][base_cols], y[vi]), use_best_model=True)
        oof[vi] = m.predict_proba(Xbase.iloc[vi][base_cols])[:, 1]
        pte += m.predict_proba(Xte_base[k][base_cols])[:, 1] / N_FOLDS
    return oof, pte


RUN = {
    "lgbn": lambda sd: run_lgb(sd, LGBN_EXTRA, Xthin, Xte_thin, thin_cols),
    "dart": lambda sd: run_lgb(sd, DART_EXTRA, Xbase, Xte_base, base_cols),
    "cat": lambda sd: run_cat(sd),
}

for fam, seeds in NEW.items():
    for sd in seeds:
        po, pt = P(f"oof_{fam}_s{sd}.npy"), P(f"te_{fam}_s{sd}.npy")
        if os.path.exists(po) and os.path.exists(pt):
            print(f"  [{fam}] seed={sd} 已存在，跳过")
            continue
        oof, pte = RUN[fam](sd)
        _, tp = top1_f1(y, oof)
        np.save(po, R(oof))
        np.save(pt, R(pte))
        print(f"  [{fam}] seed={sd} OOF {tp}/3000   {time.time()-t0:.0f}s")


# ---------------- 按实例数加权合并 ----------------
def merge(prefix, fam):
    n_old, old_name = OLD[fam]
    parts = [n_old * R(np.load(P(f"{prefix}_{old_name}.npy")))]
    n = n_old
    for sd in NEW[fam]:
        p = P(f"{prefix}_{fam}_s{sd}.npy")
        if os.path.exists(p):
            parts.append(R(np.load(p)))
            n += 1
    return sum(parts) / n, n


fam_te, fam_oof, counts = {}, {}, {}
for fam in ("dart", "lgbn", "cat"):
    fam_te[fam], counts[fam] = merge("te", fam)
    fam_oof[fam], _ = merge("oof", fam)
print("\n合并后实例数: " + "  ".join(f"{k} {v} 个" for k, v in counts.items()))

_, tp_new = top1_f1(y, R(np.mean([R(fam_oof[f]) for f in fam_oof], axis=0)))
_, tp_old = top1_f1(y, R(np.mean(
    [R(np.load(P(f"oof_{OLD[f][1]}.npy"))) for f in ("dart", "lgbn", "cat")],
    axis=0)))
print(f"三族 OOF: 原 {tp_old}/3000 -> 多 seed 后 {tp_new}/3000"
      f"（{tp_new - tp_old:+d}）")

# ---------------- 生成提交文件 ----------------
sil = pd.read_parquet(P("silence_feat.parquet"), columns=["card_no", "sil_days"])
sil["card_no"] = sil["card_no"].astype(str)
hard = te.merge(sil, on="card_no", how="left")["sil_days"].values >= 32

score = R(np.mean([R(fam_te[f]) for f in fam_te], axis=0))
old_score = R(np.mean(
    [R(np.load(P(f"te_{OLD[f][1]}.npy"))) for f in ("dart", "lgbn", "cat")],
    axis=0))
if hard.any():
    score[hard] = 1.0
    old_score[hard] = 1.0

k = int(len(te) * TOP_FRAC)
swap = k - len(set(np.argsort(-score)[:k]) & set(np.argsort(-old_score)[:k]))
print(f"top{k} 相对原三族换动 {swap} 张")

sub = te[["card_no"]].copy()
sub["score"] = score
assert (sub["card_no"].values == te["card_no"].values).all()
out = os.path.join(PRED_DIR, "prediction_resultB_moreseeds.csv")
sub.to_csv(out, index=False)
print(f"已写出 {out}   {sub.shape}")
print(f"\n全部完成 {time.time()-t0:.0f}s")
