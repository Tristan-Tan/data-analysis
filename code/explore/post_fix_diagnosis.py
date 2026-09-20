# -*- coding: utf-8 -*-
"""组内分位修复后的二次诊断（两段，均不消耗提交次数）

【段一】逐特征 PSI × gain 风险分
  上一轮 testb_drift_check.py 的联合对抗模型被 (weekday + sil_days) mod 7
  这类组合规则主导（AUC=1.0 是数学必然），掩盖了单个特征的真实漂移程度。
  本段改用逐特征 PSI（群体稳定性指数）衡量分布偏移，再乘以从已训练模型
  导出的 gain 占比，直接排出「既重要又漂移」的维度。
    PSI < 0.1   分布稳定
    0.1 ~ 0.25  轻微偏移
    > 0.25      显著偏移

【段二】三族预测一致性：train OOF vs testB
  predict.py 最后做 rank 等权平均，输出接近均匀分布，比较分数分布看不出
  置信度。改为比较 dart / abthin_lgbn / cat 三族两两的 Spearman 相关与
  top1% 集合重叠度，两侧都取同一次训练的产物，可比性干净。
    · testB 侧一致性 ≈ train 侧  -> 模型在 testB 上同样稳定，剩余损失
      更可能来自个别特征漂移，继续沿 PSI 榜单定点排查
    · testB 侧一致性明显更低     -> 模型整体不稳（真实概念漂移），
      再抠单个特征收益有限，应转向时间加权 / 增强正则

前置: 已完成一轮 10_train_models.py testb（需要 model/ 与 data/interim 下的
      oof_*.npy、te_*.npy）
用法: PYTHONPATH=code python code/explore/post_fix_diagnosis.py
耗时: 约 1~2 分钟。⚠ 训练进行中不要跑，会抢 CPU
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P, MODEL_DIR, N_FOLDS, FINAL_RECIPE, TOP_FRAC
import assemble

t0 = time.time()
FAMILIES = ["dart", "abthin_lgbn", "cat"]


# ==================== 段一：PSI × gain ====================
def psi(a, b, bins=10):
    """群体稳定性指数。分箱边界取 train 侧分位数。"""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, bins=edges)[0] / len(a)
    pb = np.histogram(b, bins=edges)[0] / len(b)
    pa, pb = np.clip(pa, 1e-6, None), np.clip(pb, 1e-6, None)
    return float(((pb - pa) * np.log(pb / pa)).sum())


def dart_gain_share():
    path = os.path.join(MODEL_DIR, "feature_cols.json")
    with open(path) as f:
        mcols = json.load(f)
    gain, n = np.zeros(len(mcols)), 0
    for sd in FINAL_RECIPE["dart"]:
        for k in range(N_FOLDS):
            p = os.path.join(MODEL_DIR, f"dart_s{sd}_f{k}.txt")
            if not os.path.exists(p):
                continue
            g = lgb.Booster(model_file=p).feature_importance("gain")
            if len(g) == len(mcols):
                gain += g
                n += 1
    if n == 0:
        return None
    s = pd.Series(gain / n, index=mcols)
    return s / s.sum()


print("=" * 70)
print("【段一】逐特征 PSI × gain 风险分")
tr, Xtr, y, cols = assemble.build_train()
te, Xte_list = assemble.build_test_folds(N_FOLDS, cols)
Xte = Xte_list[0]
print(f"  矩阵就绪 train{Xtr.shape} / testB{Xte.shape}   {time.time()-t0:.0f}s")

gshare = dart_gain_share()
if gshare is None:
    print("  ⚠ 没读到 dart 模型，gain 置 0，只看纯 PSI")
    gshare = pd.Series(0.0, index=cols)

rows = []
for c in cols:
    a = Xtr[c].to_numpy(dtype=np.float64)
    b = Xte[c].to_numpy(dtype=np.float64)
    rows.append({"feature": c,
                 "psi": psi(a, b),
                 "gain_share": float(gshare.get(c, 0.0)),
                 "train_nan": float(np.isnan(a).mean()),
                 "testb_nan": float(np.isnan(b).mean()),
                 "train_mean": float(np.nanmean(a)),
                 "testb_mean": float(np.nanmean(b))})
d = pd.DataFrame(rows)
d["risk"] = d["psi"] * d["gain_share"]

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda v: f"{v:.6f}")

print("\n-- 风险分 Top25（risk = psi × gain_share，最该优先处理的）--")
print(d.sort_values("risk", ascending=False).head(25)[
    ["feature", "risk", "psi", "gain_share", "train_mean", "testb_mean"]
].to_string(index=False))

print("\n-- 纯 PSI Top20（漂移最大，不论重要性）--")
print(d.sort_values("psi", ascending=False).head(20)[
    ["feature", "psi", "gain_share", "train_mean", "testb_mean"]
].to_string(index=False))

nan_gap = d.assign(nan_gap=(d["testb_nan"] - d["train_nan"]).abs())
nan_gap = nan_gap[nan_gap["nan_gap"] > 0.01].sort_values("nan_gap", ascending=False)
print(f"\n-- 缺失率两侧相差 >1% 的维度（共 {len(nan_gap)} 个）--")
print(nan_gap.head(10)[["feature", "train_nan", "testb_nan", "gain_share"]]
      .to_string(index=False) if len(nan_gap) else "  （无）")

print(f"\n  PSI>0.25 的维度数: {(d['psi'] > 0.25).sum()} / {len(d)}"
      f"，其中 gain_share>0.5% 的: "
      f"{((d['psi'] > 0.25) & (d['gain_share'] > 0.005)).sum()}")


# ==================== 段二：三族一致性 ====================
print("\n" + "=" * 70)
print("【段二】三族预测一致性：train OOF vs testB")


def load_set(prefix):
    out = {}
    for m in FAMILIES:
        p = P(f"{prefix}_{m}.npy")
        if os.path.exists(p):
            out[m] = np.load(p)
    return out


def topk_overlap(a, b, k):
    ia = set(np.argsort(-a)[:k])
    ib = set(np.argsort(-b)[:k])
    return len(ia & ib) / k


oof, tev = load_set("oof"), load_set("te")
if len(oof) < 2 or len(tev) < 2:
    print("  ⚠ oof_*.npy / te_*.npy 不全，跳过本段")
else:
    k_tr = int(len(next(iter(oof.values()))) * TOP_FRAC)
    k_te = int(len(next(iter(tev.values()))) * TOP_FRAC)
    print(f"  train top{k_tr} / testB top{k_te}")
    print(f"\n{'族对':<28}{'train Spearman':>16}{'testB Spearman':>16}"
          f"{'train top1%重叠':>18}{'testB top1%重叠':>18}")
    pairs = [(a, b) for i, a in enumerate(FAMILIES) for b in FAMILIES[i + 1:]]
    for a, b in pairs:
        if a not in oof or b not in oof or a not in tev or b not in tev:
            continue
        s_tr = spearmanr(oof[a], oof[b]).correlation
        s_te = spearmanr(tev[a], tev[b]).correlation
        o_tr = topk_overlap(oof[a], oof[b], k_tr)
        o_te = topk_overlap(tev[a], tev[b], k_te)
        print(f"{a + ' vs ' + b:<28}{s_tr:>16.4f}{s_te:>16.4f}"
              f"{o_tr:>18.4f}{o_te:>18.4f}")

    print("\n  判读：testB 侧的 top1% 重叠度若明显低于 train 侧（比如低 0.1 以上），"
          "\n        说明三族在 testB 头部判断上分歧变大 —— 属于模型整体不稳，"
          "\n        再抠单个特征收益有限，应转向时间加权 / 增强正则。")

print(f"\n全部完成 {time.time()-t0:.0f}s")
