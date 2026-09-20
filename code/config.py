# -*- coding: utf-8 -*-
"""全局配置：所有脚本共用。评审只需确认本文件的路径与配方即可。

【赛段隔离 —— 本文件最重要的设计】
初赛 A 榜与复赛 B 榜**不共用任何中间产物**：

    data/interim_A/   model/A/     ← A 榜（401/408 维，池子里只有 train+testA）
    data/interim_B/   model/B/     ← B 榜（402/409 维，池子里含 testB）

原因：`01_parse_raw.py` 在 testB 存在时会把它并入同一份 txn/card/cst
（跨卡图特征必须共用参考池，见说明文档 §一.5）。但这会**改变 A 榜的特征
取值** —— `occupation_freq` 等 6 个频次编码、对手网络度、exp5 的低度数变体
全都依赖"池子里有哪些卡"。A 榜成绩是在 testB 尚不存在时跑出的，若在 testB
已落盘的环境里重跑 A 榜链路而不做隔离，得到的将是另一套特征。

因此赛段由环境变量 `FRAUD_ROUND` 显式指定，**不设默认值**：历史上
`10_train_models.py` 因为 TARGET 默认 testa，曾静默跑错 2.5 小时。
请用 `code/scripts/run_train.sh A|B`，或自行 `export FRAUD_ROUND=B`。
"""
import os

# ---------------- 赛段 ----------------
ROUND = os.environ.get("FRAUD_ROUND", "").strip().upper()
if ROUND not in ("A", "B"):
    raise SystemExit(
        "\n环境变量 FRAUD_ROUND 未设置或取值非法（当前值: "
        f"{os.environ.get('FRAUD_ROUND')!r}）。\n"
        "  A = 初赛（testA，401/408 维，池中不含 testB）\n"
        "  B = 复赛（testB，402/409 维，池中含 testB）\n"
        "请使用 bash code/scripts/run_train.sh A|B，\n"
        "或手动 export FRAUD_ROUND=B 后再运行单个脚本。\n"
        "本项目刻意不设默认值 —— 跑错赛段会静默浪费 2.5 小时。")

# ---------------- 路径 ----------------
# 默认取本文件上两级（即项目根目录）；也可用环境变量 FRAUD_BASE 覆盖
BASE = os.environ.get(
    "FRAUD_BASE",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA = os.path.join(BASE, "data")
TRAIN_DIR = os.path.join(DATA, "train")      # *_train_testa.csv + train.csv
TESTA_DIR = os.path.join(DATA, "testA")      # testa.csv
TESTB_DIR = os.path.join(DATA, "testB")      # 复赛 5 个文件
INTERIM = os.path.join(DATA, f"interim_{ROUND}")
MODEL_DIR = os.path.join(BASE, "model", ROUND)
PRED_DIR = os.path.join(BASE, "prediction_result")

for _d in (INTERIM, MODEL_DIR, PRED_DIR):
    os.makedirs(_d, exist_ok=True)


def P(name):
    """中间产物路径（已按赛段隔离）"""
    return os.path.join(INTERIM, name)


def M(name):
    """模型产物路径（已按赛段隔离）"""
    return os.path.join(MODEL_DIR, name)


# ---------------- 赛段规格 ----------------
# include_testb : 01_parse_raw 是否把 testB 并入统一参考池
# use_smy_te    : 是否启用 07b 的 smy_cd 目标编码（A 榜最终提交未包含）
# blend         : 最终融合的成员列表，**重复即表示权重**
#                 （融合公式 R(mean([R(x) for x in members]))，见 predict.py）
ROUND_SPEC = {
    "A": dict(
        target="testa",
        include_testb=False,
        use_smy_te=False,
        blend=["dart", "abthin_lgbn", "cat"],
        out_name="prediction_resultA.csv",
        score="0.917771（692/754）",
    ),
    "B": dict(
        target="testb",
        include_testb=True,
        use_smy_te=True,
        # 复赛最终配方：nosil_partial 占 2 份权重、nosil_full 占 1 份
        blend=["dart", "abthin_lgbn", "cat",
               "nosil_partial", "nosil_partial", "nosil_full"],
        out_name="prediction_resultB.csv",
        score="0.880637（664/754）",
    ),
}
SPEC = ROUND_SPEC[ROUND]
TARGET = SPEC["target"]

# ---------------- 复现常量 ----------------
# ⚠ 折划分种子固定为 42：exp5 暴露编码是在该折下 fold-safe 生成的，
#   换折种子会导致验证折样本的编码包含自身 label（泄露）。
FOLD_SEED = 42
N_FOLDS = 5

# 提交口径：testA / testB 均为 75,400 张卡，取前 1% = 754 张判为欺诈
TOP_FRAC = 0.01

# 硬规则阈值：train 中 sil_days >= 32 的 8 张卡全部为欺诈（见说明文档 §一.7）
HARD_SIL_DAYS = 32

# 三个基础模型族及其随机种子。lgbn 键对应 AB-thin 窄树。
FINAL_RECIPE = {
    "dart": [42],
    "lgbn": [42, 2026],
    "cat":  [42],
}

# no-silence 变体（仅 B 榜使用，见 11_train_nosil.py）
NOSIL_SEED = 42
# partial 模式屏蔽的 13 维：绝对尺度量 + 交互项（PSI 0.25~2.6，跨批次不可比）
NOSIL_DROP_PARTIAL = [
    "sil_days", "sil_hours", "first_sil_days", "sil_div_span",
    "ntxn_x_sil", "ntxn_div_sil", "insum_x_sil", "bal_x_sil",
    "t1share_x_sil", "t3share_x_sil", "netflow_x_sil",
    "retention_x_sil", "txnrate_x_sil",
]

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


def banner(script):
    """每个脚本开头打印赛段，避免跑错赛段而不自知"""
    print(f"===== [{script}] 赛段 {ROUND}（目标测试集 {TARGET}，"
          f"{'含' if SPEC['include_testb'] else '不含'} testB 参考池，"
          f"smy_te {'启用' if SPEC['use_smy_te'] else '停用'}）=====")
    print(f"      中间产物 {INTERIM}")
    print(f"      模型目录 {MODEL_DIR}")
