# -*- coding: utf-8 -*-
"""特征矩阵组装 —— 训练与预测**共用同一份代码**，杜绝两侧口径漂移。

402 维构成：
    299  features_v4        窗口聚合 / 尾部行为 / smy 与渠道展开 / 静态表
     56  exp5 暴露编码       8 变体 × 7 统计（fold-safe）
     19  inflow_feat        入金侧 label-free（发送方扇入度等）
      5  财富归一            见下方公式，v12 的胜负手
     22  silence 族          本方案核心创新
      1  smy_te             smy_cd 加权目标编码（fold-safe/fold-matched，
                             同 exp5 口径；单变量 AUC 0.7714，与已有特征
                             最高相关仅 0.4767，OOF 快速验证 +8 命中）
"""
import gc
import os

import numpy as np
import pandas as pd

from config import P, SIL_COLS


def _wealth(d):
    """财富归一 5 维。分母 abs(...)+100 防零防负。
    ⚠ in_vs_aum / bal_vs_aum 的分母是 **mo_da_aum_val（月日均）**，
      inmax_vs_aum 的分子是 **in_max2（来自 inflow_feat）**、分母是 aum_val。
      自检：in_vs_aum 在 label=1 上均值应 ≈178，label=0 上 ≈21。"""
    d["in_vs_aum"] = d["in_sum"] / (d["mo_da_aum_val"].abs() + 100)
    d["bal_vs_aum"] = d["bal_last"] / (d["mo_da_aum_val"].abs() + 100)
    d["inmax_vs_aum"] = d["in_max2"] / (d["aum_val"].abs() + 100)
    d["in_retention"] = d["bal_last"] / (d["in_sum"] + 1)
    d["sender_per_in"] = d["n_in_senders"] / (d["n_in"] + 1)
    return d


def load_parts():
    feat = pd.read_parquet(P("features_v4.parquet"))
    infeat = pd.read_parquet(P("inflow_feat.parquet"))
    sil = pd.read_parquet(P("silence_feat.parquet"),
                          columns=["card_no"] + SIL_COLS)
    sil = sil.rename(columns={c: "S_" + c for c in SIL_COLS})
    for d in (feat, infeat, sil):
        d["card_no"] = d["card_no"].astype(str)
    return feat, infeat, sil


def build(keys, enc, feat, infeat, sil, cols=None):
    """keys: 只含 card_no 的 DataFrame；enc: 该侧的 exp5 编码
    返回 (X, cols)。cols 传入时按给定顺序取列，保证 train/test 完全一致。"""
    ec = [c for c in enc.columns if c != "card_no"]
    d = (keys.merge(feat, on="card_no", how="left")
             .merge(enc, on="card_no", how="left")
             .merge(infeat, on="card_no", how="left")
             .merge(sil, on="card_no", how="left"))
    d[ec] = d[ec].fillna(0)          # 暴露编码缺失 = 无暴露，填 0
    d = _wealth(d)
    if cols is None:
        nc = ([c for c in infeat.columns if c != "card_no"]
              + ["in_vs_aum", "bal_vs_aum", "inmax_vs_aum",
                 "in_retention", "sender_per_in"])
        sc = [c for c in d.columns if c.startswith("S_")]
        cols = [c for c in feat.columns if c != "card_no"] + ec + nc + sc
    X = d[cols].astype(np.float32)
    del d
    gc.collect()
    return X, cols


def _merge_smy_te(enc, target, k=None):
    """把 07b 生成的 smy_te（fold-safe/fold-matched）合并进 enc，
    复用 enc 已有的 'ec = exp5 列' 通用处理逻辑（merge / fillna(0)）。
    target: 'train' | 'testa' | 'testb'；k 仅测试侧使用（第几折）。"""
    if target == "train":
        smy_te = pd.read_parquet(P("smy_te_tr.parquet"))
    else:
        pfx = "smy_te_te" if target == "testa" else "smy_te_teb"
        p = P(f"{pfx}_f{k}.parquet")
        assert os.path.exists(p), f"缺少 smy_te fold-matched 编码 {p}，请先跑 07b"
        smy_te = pd.read_parquet(p)
    smy_te["card_no"] = smy_te["card_no"].astype(str)
    return enc.merge(smy_te, on="card_no", how="left")


def build_train(cols=None):
    tr = pd.read_parquet(P("train.parquet"))
    tr["card_no"] = tr["card_no"].astype(str)
    y = tr["label"].astype(float).astype(np.int8).values
    enc = pd.read_parquet(P("exp5_tr.parquet"))
    enc["card_no"] = enc["card_no"].astype(str)
    enc = _merge_smy_te(enc, "train")
    feat, infeat, sil = load_parts()
    X, cols = build(tr[["card_no"]], enc, feat, infeat, sil, cols)
    return tr, X, y, cols


def build_test_folds(target, n_folds, cols):
    """返回 5 份测试矩阵，第 k 份使用 fold-matched 的 exp5 + smy_te 编码。
    target: 'testa' | 'testb'"""
    te = pd.read_parquet(P(f"{target}.parquet"))
    te["card_no"] = te["card_no"].astype(str)
    feat, infeat, sil = load_parts()
    pfx = "exp5_te" if target == "testa" else "exp5_teb"
    Xs = []
    for k in range(n_folds):
        p = P(f"{pfx}_f{k}.parquet")
        assert os.path.exists(p), f"缺少 fold-matched 编码 {p}，请先跑 07"
        enc = pd.read_parquet(p)
        enc["card_no"] = enc["card_no"].astype(str)
        enc = _merge_smy_te(enc, target, k)
        X, _ = build(te[["card_no"]], enc, feat, infeat, sil, cols)
        Xs.append(X)
    return te, Xs
