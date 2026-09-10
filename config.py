# -*- coding: utf-8 -*-
"""全局配置：所有脚本共用。评审只需确认本文件的路径即可。"""
import os

# ---------------- 路径 ----------------
# 默认取本文件上两级（即项目根目录）；也可用环境变量 FRAUD_BASE 覆盖
BASE = os.environ.get(
    "FRAUD_BASE",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA = os.path.join(BASE, "data")
TRAIN_DIR = os.path.join(DATA, "train")     # 放 *_train_testa.csv + train.csv
TESTA_DIR = os.path.join(DATA, "testA")     # 放 testa.csv
TESTB_DIR = os.path.join(DATA, "testB")     # 复赛用
INTERIM = os.path.join(DATA, "interim")     # 全部中间 parquet
MODEL_DIR = os.path.join(BASE, "model")
PRED_DIR = os.path.join(BASE, "prediction_result")

for _d in (INTERIM, MODEL_DIR, PRED_DIR):
    os.makedirs(_d, exist_ok=True)


def P(name):
    """中间产物路径"""
    return os.path.join(INTERIM, name)


# ---------------- 复现常量 ----------------
# ⚠ 折划分种子固定为 42：exp5 暴露编码是在该折下 fold-safe 生成的，
#   换折种子会导致验证折样本的编码包含自身 label（泄露）。
FOLD_SEED = 42
N_FOLDS = 5

# 提交口径：testa 共 75,400 张卡，取前 1% = 754 张判为欺诈
TOP_FRAC = 0.01

# silence 特征锚点：train+testA 全量末笔日期的最大值
# 复赛若 testb 末笔日期分布发生平移，必须重新确认（见说明文档 §三）
SILENCE_ANCHOR = "2026-01-01"

# 最终提交所用的三个模型族及其随机种子（严格对应 abthin_w100）。
# lgbn 键表示 408 维 abthin_lgbn；DART/CatBoost 仍使用基础 401 维。
FINAL_RECIPE = {
    "dart": [42],
    "lgbn": [42, 2026],
    "cat":  [42],
}

# ---------------- 模型超参 ----------------
LGB_BASE = dict(objective="binary", metric="auc", learning_rate=0.05,
                num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                n_jobs=5, verbosity=-1)

# dart：训练时随机丢弃已有树，强正则
DART_EXTRA = dict(boosting="dart", drop_rate=0.1, skip_drop=0.5,
                  learning_rate=0.06)

# lgbn：窄树 + 低学习率 + 低特征采样
LGBN_EXTRA = dict(num_leaves=15, learning_rate=0.03,
                  min_data_in_leaf=100, feature_fraction=0.6)

CAT_PARAMS = dict(iterations=3000, learning_rate=0.05, depth=6,
                  l2_leaf_reg=3, loss_function="Logloss", eval_metric="AUC",
                  thread_count=5, od_type="Iter", od_wait=100, verbose=False)

# ---------------- silence 特征列名 ----------------
SIL_COLS = [
    "sil_days", "sil_hours", "sil_beyond_norm", "first_sil_days",
    "last_hour", "last_wd", "last_is_bizhour", "last_is_deepnight",
    "ntxn_x_sil", "ntxn_div_sil", "insum_x_sil", "bal_x_sil",
    "t1share_x_sil", "t3share_x_sil", "netflow_x_sil", "retention_x_sil",
    "txnrate_x_sil", "sil_div_span",
    "sil_pct_in_ntxn_bin", "ntxn_pct_in_sil_bin",
    "bal_pct_in_sil_bin", "insum_pct_in_sil_bin",
]
SIL_HELPER = ["n_txn", "in_sum", "out_sum", "bal_last", "span_days",
              "n_t1", "n_t3", "n_t7", "last_is_in"]


def top1_f1(y, score):
    """线下评估口径：与线上完全一致（前 1% 判正，算 F1）"""
    import numpy as np
    k = int(len(y) * TOP_FRAC)
    tp = int(np.asarray(y)[np.argsort(-score)[:k]].sum())
    p, r = tp / k, tp / np.asarray(y).sum()
    return 2 * p * r / (p + r + 1e-9), tp
