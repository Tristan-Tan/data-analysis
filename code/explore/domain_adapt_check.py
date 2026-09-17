# -*- coding: utf-8 -*-
"""域适应可行性验证：importance weighting 能不能用

【背景】
testB 无标签，"针对 B 榜单独训练"只能走域适应。其中唯一有理论依据的是
样本重加权：用对抗模型算出每个 train 样本「像 testB 的程度」p(testB|x)，
取 w = p/(1-p) 作训练权重，让模型更关注像 testB 的那部分 train 样本。

【必须先验证的前提】
协变量偏移的数学假设要求两个分布有**重叠支撑集**。实测全量特征下对抗
AUC = 1.0000（完全可分），此时 p 对所有 train 样本趋近 0、权重全塌成 0，
方法直接失效。但那个 1.0 是人工规则造成的：sil_days = 锚点 - 末笔日，而
两批锚点星期几不同（train 2026-01-01 周四 / testB 2026-01-31 周六），
使 (weekday + sil_days) mod 7 成为 100% 精确的判别规则。

因此本脚本剔除「绝对日历量 + sil 绝对尺度量与交互项」后重算对抗 AUC：
  · AUC < 0.9  -> 存在重叠支撑集，importance weighting 可行
  · AUC >= 0.95 -> 两分布几乎不重叠，域适应在数学上不成立，应放弃

【通过后还要看两个诊断】
  · ESS/N（有效样本量占比）：< 0.2 说明权重过度集中，重训会过拟合
  · 正/负样本的平均权重对比：若正样本权重被系统性压低，重加权等于削弱
    本就稀缺的 3000 个欺诈样本，得不偿失

用法: PYTHONPATH=code python code/explore/domain_adapt_check.py
耗时: 约 5 分钟
输出: data/interim/da_weights.npy（验证通过时才写）
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P, N_FOLDS, LGB_BASE
import assemble

t0 = time.time()

# 能重构出 (weekday + sil_days) mod 7 规则的维度，以及 PSI 0.25~2.6 的
# sil 绝对尺度量与交互项。保留组内分位（PSI 0.017/0.006，跨批次稳定）。
DROP = [
    "anchor_weekday", "anchor_hour", "last_hour",
    "S_last_wd", "S_last_hour",
    "S_sil_days", "S_sil_hours", "S_first_sil_days", "S_sil_div_span",
    "S_ntxn_x_sil", "S_ntxn_div_sil", "S_insum_x_sil", "S_bal_x_sil",
    "S_t1share_x_sil", "S_t3share_x_sil", "S_netflow_x_sil",
    "S_retention_x_sil", "S_txnrate_x_sil",
]

tr, Xtr, y, cols = assemble.build_train()
te, Xte_list = assemble.build_test_folds("testb", N_FOLDS, cols)
Xte = Xte_list[0]
use_cols = [c for c in cols if c not in DROP]
print(f"特征 {len(cols)} 维 -> 剔除 {len(cols) - len(use_cols)} 维后 "
      f"{len(use_cols)} 维   {time.time()-t0:.0f}s")

# ---------------- 对抗验证 ----------------
X = pd.concat([Xtr[use_cols], Xte[use_cols]], ignore_index=True)
y_adv = np.r_[np.zeros(len(Xtr), dtype=np.int8),
              np.ones(len(Xte), dtype=np.int8)]
p_oof = np.zeros(len(X))
for k, (ti, vi) in enumerate(StratifiedKFold(
        5, shuffle=True, random_state=42).split(X, y_adv)):
    m = lgb.train(dict(LGB_BASE, seed=42),
                  lgb.Dataset(X.iloc[ti], y_adv[ti]), 1000,
                  valid_sets=[lgb.Dataset(X.iloc[vi], y_adv[vi])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    p_oof[vi] = m.predict(X.iloc[vi], num_iteration=m.best_iteration)
    print(f"  fold{k} best_iter={m.best_iteration}   {time.time()-t0:.0f}s")

auc = roc_auc_score(y_adv, p_oof)
print("\n" + "=" * 70)
print(f"【剔除人工规则维度后的对抗 AUC】= {auc:.4f}（原全量特征下为 1.0000）")

if auc >= 0.95:
    print("  => 两分布几乎仍不重叠，协变量偏移假设不成立，"
          "importance weighting 不可行，应放弃域适应路线。")
    sys.exit(0)
print("  => 存在重叠支撑集，继续评估权重可用性")

# ---------------- 权重与诊断 ----------------
p = np.clip(p_oof[:len(Xtr)], 1e-4, 1 - 1e-4)
w = p / (1 - p)
lo, hi = np.quantile(w, [0.01, 0.99])
w = np.clip(w, lo, hi)
w = w / w.mean()

ess = w.sum() ** 2 / (w ** 2).sum()
print(f"\n【权重诊断】")
print(f"  ESS/N = {ess / len(w):.3f}"
      f"（有效样本量 {ess:,.0f} / {len(w):,}；<0.2 则权重过度集中）")
print(f"  权重分位 p05/p50/p95 = "
      f"{np.quantile(w, 0.05):.3f} / {np.quantile(w, 0.5):.3f} / "
      f"{np.quantile(w, 0.95):.3f}")
w1, w0 = w[y == 1].mean(), w[y == 0].mean()
print(f"  正样本平均权重 {w1:.3f} vs 负样本 {w0:.3f}   比值 {w1 / w0:.3f}")
if w1 / w0 < 0.8:
    print("  ⚠ 正样本权重被系统性压低 —— 重加权会削弱本就稀缺的 3000 个欺诈"
          "样本，风险高于收益")
elif w1 / w0 > 1.2:
    print("  正样本权重反而更高，对不平衡问题是额外利好")
else:
    print("  正负样本权重基本均衡，无系统性偏向")

np.save(P("da_weights.npy"), w)
print(f"\n权重已保存到 {P('da_weights.npy')}")
print(f"""
【结论判读】
  同时满足 AUC<0.9、ESS/N>=0.2、正负权重比 >=0.8 -> 值得用重加权重训一个
  lgbn 族验证（约 15 分钟）；任一条不满足则放弃，把时间留给多 seed 与材料。
""")
print(f"全部完成 {time.time()-t0:.0f}s")
