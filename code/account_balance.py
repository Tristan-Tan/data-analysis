# -*- coding: utf-8 -*-
"""最终 AB-thin 账户/余额特征。

特征只使用单卡自身的流水和静态账户归属，不使用 label。时序余额变化先在
``(card_no, cst_accno)`` 内计算，再聚合到卡粒度，避免一卡多账户串线。

该模块同时供训练和预测调用；若缓存已存在则直接复用。最终模型只取 7 个
预先固定的特征，列名与 A 榜 ``abthin_w100`` 完全一致。
"""
import os

import numpy as np
import pandas as pd
import polars as pl

from config import P


FEATURE_FILE = "account_balance_v1.parquet"

# 资金分布 4 维 + 余额一致性 3 维。顺序不可改变。
THIN_COLS = [
    "AB_acc_in_hhi",
    "AB_primary_in_share",
    "AB_acc_out_hhi",
    "AB_primary_out_share",
    "AB_relresid_after_mean",
    "AB_acc_resid_best_p90_mean",
    "AB_resid_after_p90",
]


def create_features():
    """从 txn/card parquet 构造 7 维 AB-thin 特征并落盘。"""
    txn = pl.read_parquet(
        P("txn.parquet"),
        columns=[
            "card_no", "cst_accno", "accno_txn_sn", "tms", "acct_bal",
            "inflow_amt", "outflow_amt",
        ],
    )
    card = pl.read_parquet(
        P("card.parquet"), columns=["card_no", "prim_cst_accno"])
    txn = txn.join(card, on="card_no", how="left")
    account_keys = ["card_no", "cst_accno"]

    # 同秒多笔由账户交易序号确定顺序。
    txn = txn.sort(["card_no", "cst_accno", "tms", "accno_txn_sn"])
    txn = txn.with_columns([
        (pl.col("inflow_amt") - pl.col("outflow_amt")).alias("_signed_amt"),
        (pl.col("cst_accno") == pl.col("prim_cst_accno"))
        .fill_null(False).alias("_is_primary"),
        pl.col("acct_bal").shift(1).over(account_keys).alias("_prev_bal"),
        pl.col("acct_bal").shift(-1).over(account_keys).alias("_next_bal"),
    ])
    txn = txn.with_columns([
        ((pl.col("acct_bal") - pl.col("_prev_bal")) - pl.col("_signed_amt"))
        .abs().alias("_abs_resid_after"),
        ((pl.col("_next_bal") - pl.col("acct_bal")) - pl.col("_signed_amt"))
        .abs().alias("_abs_resid_before"),
    ])
    txn = txn.with_columns(
        (pl.col("_abs_resid_after") /
         (pl.col("_signed_amt").abs() + 1)).alias("_rel_resid_after")
    )

    # 账户画像：先账户内统计，再计算该账户在卡内的资金占比。
    acc = txn.group_by(account_keys).agg([
        pl.col("inflow_amt").sum().alias("_insum"),
        pl.col("outflow_amt").sum().alias("_outsum"),
        pl.col("_is_primary").max().alias("_primary"),
        pl.col("_abs_resid_after").quantile(0.9).alias("_ra_p90"),
        pl.col("_abs_resid_before").quantile(0.9).alias("_rb_p90"),
    ])
    acc = acc.with_columns([
        pl.min_horizontal("_ra_p90", "_rb_p90").alias("_resid_best_p90"),
        (pl.col("_insum") /
         (pl.col("_insum").sum().over("card_no") + 1)).alias("_in_share"),
        (pl.col("_outsum") /
         (pl.col("_outsum").sum().over("card_no") + 1)).alias("_out_share"),
    ])

    acc_card = acc.group_by("card_no").agg([
        (pl.col("_in_share") ** 2).sum().alias("AB_acc_in_hhi"),
        pl.col("_in_share").filter(pl.col("_primary")).sum()
        .alias("AB_primary_in_share"),
        (pl.col("_out_share") ** 2).sum().alias("AB_acc_out_hhi"),
        pl.col("_out_share").filter(pl.col("_primary")).sum()
        .alias("AB_primary_out_share"),
        pl.col("_resid_best_p90").mean()
        .alias("AB_acc_resid_best_p90_mean"),
    ])
    balance_card = txn.group_by("card_no").agg([
        pl.col("_rel_resid_after").mean().alias("AB_relresid_after_mean"),
        pl.col("_abs_resid_after").quantile(0.9)
        .alias("AB_resid_after_p90"),
    ])
    output = acc_card.join(balance_card, on="card_no", how="left")
    output = output.select(["card_no"] + THIN_COLS)
    output.write_parquet(P(FEATURE_FILE))
    return output.to_pandas()


def load_or_create_features():
    path = P(FEATURE_FILE)
    features = pd.read_parquet(path) if os.path.exists(path) else create_features()
    features["card_no"] = features["card_no"].astype(str)
    assert features["card_no"].is_unique, "账户特征 card_no 不唯一"
    missing = [col for col in THIN_COLS if col not in features.columns]
    assert not missing, f"账户特征缓存缺列：{missing}"
    return features[["card_no"] + THIN_COLS].copy()


def append_features(X, keys, features):
    """按 card_no 一一对齐后一次追加，避免 DataFrame 碎片化。"""
    aligned = keys[["card_no"]].copy()
    aligned["card_no"] = aligned["card_no"].astype(str)
    aligned = aligned.merge(
        features, on="card_no", how="left", validate="one_to_one")
    values = aligned[THIN_COLS].replace([np.inf, -np.inf], np.nan)
    values = values.astype(np.float32)
    values.index = X.index
    return pd.concat([X, values], axis=1)
