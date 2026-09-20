# -*- coding: utf-8 -*-
"""B 榜实验公共模块：独立输出目录、带校验的缓存、换入/换出诊断

设计约束（来自实验要求）：
  · 所有产物写入 data/interim_labB 与 prediction_result/labB，
    **不触碰** data/interim、model/、prediction_result 根目录下的正式产物
  · 缓存不能"文件存在就复用"：manifest 必须核对输入数据版本、目标集、
    卡号集合指纹、特征定义版本，任一不符即重算并打印差异
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE, DATA, P, PRED_DIR, TOP_FRAC

LAB_INTERIM = os.path.join(DATA, "interim_labB")
LAB_PRED = os.path.join(PRED_DIR, "labB")
for _d in (LAB_INTERIM, LAB_PRED):
    os.makedirs(_d, exist_ok=True)


def LP(name):
    """实验中间产物路径（与正式 data/interim 完全隔离）"""
    return os.path.join(LAB_INTERIM, name)


# ---------------- 缓存校验 ----------------
def _card_fingerprint(path):
    s = pd.read_parquet(path, columns=["card_no"])["card_no"].astype(str)
    v = pd.util.hash_pandas_object(s.sort_values(), index=False).values
    return hashlib.md5(v.tobytes()).hexdigest()[:16]


def data_manifest(target, feature_version, extra=None):
    """数据版本 + 目标集 + 卡号指纹 + 特征定义版本"""
    m = {"target": target, "feature_version": feature_version}
    for n in ("txn.parquet", "train.parquet", f"{target}.parquet"):
        st = os.stat(P(n))
        m[f"src::{n}"] = {"size": st.st_size, "mtime": int(st.st_mtime)}
    m["cards::train"] = _card_fingerprint(P("train.parquet"))
    m[f"cards::{target}"] = _card_fingerprint(P(f"{target}.parquet"))
    if extra:
        m.update({f"param::{k}": v for k, v in extra.items()})
    return m


def load_cached(name, manifest):
    """manifest 完全一致才复用；不一致时打印差异并返回 None"""
    p, mp = LP(name), LP(name + ".manifest.json")
    if not (os.path.exists(p) and os.path.exists(mp)):
        return None
    with open(mp, encoding="utf-8") as f:
        old = json.load(f)
    if old == manifest:
        print(f"  [缓存命中] {name}（manifest 校验通过）")
        return pd.read_parquet(p)
    print(f"  [缓存失效] {name}，差异如下，将重算：")
    for k in sorted(set(old) | set(manifest)):
        if old.get(k) != manifest.get(k):
            print(f"    {k}: 缓存={old.get(k)} -> 当前={manifest.get(k)}")
    return None


def save_cached(df, name, manifest):
    df.to_parquet(LP(name), index=False)
    with open(LP(name + ".manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  [已落盘] {name}  {df.shape}")


# ---------------- 评估与诊断 ----------------
def R(v):
    return rankdata(v) / len(v)


def topk_hit(y, score, frac=TOP_FRAC):
    k = int(len(y) * frac)
    idx = np.argsort(-score)[:k]
    return int(np.asarray(y)[idx].sum()), k


def swap_report(base, new, y=None, frac=TOP_FRAC, label=""):
    """换入/换出分析。m 指**每侧**数量（换入 m 张、换出 m 张，两者必然相等）"""
    k = int(len(base) * frac)
    b = np.argsort(-base)[:k]
    n = np.argsort(-new)[:k]
    sb, sn = set(b.tolist()), set(n.tolist())
    swap_in = sorted(sn - sb)
    swap_out = sorted(sb - sn)
    res = {"k": k, "m_each_side": len(swap_in), "overlap": len(sb & sn)}
    if y is not None:
        y = np.asarray(y)
        res["in_pos"] = int(y[swap_in].sum()) if swap_in else 0
        res["out_pos"] = int(y[swap_out].sum()) if swap_out else 0
        res["net_pos"] = res["in_pos"] - res["out_pos"]
    if swap_in:
        # 换入卡在**基线**里的原排名（1 = 基线最高分）
        br = rankdata(-base, method="ordinal")
        r = br[swap_in]
        res["in_base_rank_min"] = int(r.min())
        res["in_base_rank_p50"] = int(np.median(r))
        res["in_base_rank_max"] = int(r.max())
    print(f"\n  [换卡分析{(' ' + label) if label else ''}] "
          f"top{res['k']}  每侧 m={res['m_each_side']}  重叠={res['overlap']}")
    if y is not None:
        print(f"    换入正例 {res['in_pos']} / 换出正例 {res['out_pos']}"
              f" -> 净收益 {res['net_pos']:+d}")
    if swap_in:
        print(f"    换入卡的基线原排名: min={res['in_base_rank_min']} "
              f"p50={res['in_base_rank_p50']} max={res['in_base_rank_max']}")
    return res, swap_in, swap_out


def boundary_auc(y, base, new, lo=500, hi=1500):
    """基线边界区域的条件区分能力：
    取基线排名落在 [lo, hi) 的卡作为子集，比较基线分数与新分数在**该子集内**
    对真实标签的 AUC。这直接回答「新信息能否改善边界判断」，
    而不是被头部易分样本稀释的全局 AUC。"""
    from sklearn.metrics import roc_auc_score
    br = rankdata(-base, method="ordinal")
    m = (br >= lo) & (br < hi)
    y = np.asarray(y)[m]
    if y.sum() == 0 or y.sum() == len(y):
        print(f"  [边界区分能力] 排名[{lo},{hi}) 子集内正例数为 {int(y.sum())}，跳过")
        return None
    a_b = roc_auc_score(y, np.asarray(base)[m])
    a_n = roc_auc_score(y, np.asarray(new)[m])
    print(f"  [边界区分能力] 基线排名[{lo},{hi}) 子集 n={m.sum()} 正例={int(y.sum())}"
          f"\n    子集内 AUC: 基线 {a_b:.4f} -> 新 {a_n:.4f}（{a_n - a_b:+.4f}）")
    return {"n": int(m.sum()), "pos": int(y.sum()), "auc_base": a_b,
            "auc_new": a_n, "delta": a_n - a_b}


def coverage_report(df, cols, name=""):
    """数据覆盖与缺失率"""
    nan = df[cols].isna().mean()
    print(f"\n  [覆盖率{(' ' + name) if name else ''}] {len(cols)} 列，"
          f"缺失率 min={nan.min():.3%} p50={nan.median():.3%} max={nan.max():.3%}")
    worst = nan.sort_values(ascending=False).head(5)
    print("    缺失率最高的 5 列: "
          + ", ".join(f"{k}={v:.1%}" for k, v in worst.items()))
    return nan
