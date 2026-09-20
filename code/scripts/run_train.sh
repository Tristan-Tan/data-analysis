#!/usr/bin/env bash
# 完整训练流程（含全部特征加工）
# 用法: bash code/scripts/run_train.sh A     # 初赛，约 2.5 小时
#       bash code/scripts/run_train.sh B     # 复赛，约 2.9 小时（多 nosil 两族）
#
# 两个赛段的中间产物与模型完全隔离：
#   A -> data/interim_A/  model/A/      （参考池不含 testB，401/408 维）
#   B -> data/interim_B/  model/B/      （参考池含 testB，402/409 维）
# 因此两榜可以任意顺序、重复运行，互不污染。
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/code:${PYTHONPATH:-}"

ROUND="${1:-}"
if [[ "$ROUND" != "A" && "$ROUND" != "B" ]]; then
    echo "用法: bash code/scripts/run_train.sh A|B" >&2
    echo "  A = 初赛（testA）   B = 复赛（testB）" >&2
    echo "本脚本刻意不设默认赛段 —— 跑错赛段会静默浪费 2.5 小时。" >&2
    exit 2
fi
export FRAUD_ROUND="$ROUND"
echo "########## 赛段 $ROUND 训练开始  $(date '+%F %T') ##########"

run () { echo; echo "===== $1 ====="; shift; python "$@"; }

run "01 原始 CSV -> parquet"          code/train/01_parse_raw.py
run "02 特征 v1（94 列）"              code/train/02_features_v1.py
run "03 特征 v2（142 列）"             code/train/03_features_v2.py
run "04 特征 v3（174 列）"             code/train/04_features_v3.py
run "05 特征 v4（300 列）"             code/train/05_features_v4.py
run "06 对手边表"                      code/train/06_edge_tables.py
run "07 exp5 暴露编码（fold-matched）" code/train/07_exp5_encoding.py
if [[ "$ROUND" == "B" ]]; then
    run "07b smy_cd 目标编码（仅 B 榜）" code/train/07b_smy_te_encoding.py
else
    echo; echo "===== 07b smy_cd 目标编码：A 榜不含该维度，跳过 ====="
fi
run "08 入金侧特征（19 维）"           code/train/08_inflow_feat.py
run "09 silence 特征族（22 维）"       code/train/09_silence_feat.py
run "10 AB-thin + 三族模型"            code/train/10_train_models.py
if [[ "$ROUND" == "B" ]]; then
    run "11 no-silence 两族（仅 B 榜）"  code/train/11_train_nosil.py
else
    echo; echo "===== 11 no-silence 两族：A 榜配方不含，跳过 ====="
fi

echo
echo "########## 赛段 $ROUND 训练完成  $(date '+%F %T') ##########"
echo "模型已落盘至 model/$ROUND/"
echo "下一步: bash code/scripts/run_predict.sh $ROUND"
