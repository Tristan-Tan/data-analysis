#!/usr/bin/env bash
# 仅预测：加载 model/{赛段}/ 下已训练好的模型，不重新训练，约 3~5 分钟
# 用法: bash code/scripts/run_predict.sh A     -> prediction_result/prediction_resultA.csv
#       bash code/scripts/run_predict.sh B     -> prediction_result/prediction_resultB.csv
#
# 前提：对应赛段的 data/interim_{A|B}/ 与 model/{A|B}/ 均已就绪。
# 若只拿到随材料提交的 model/ 而没有中间产物，请先跑 run_features.sh 补齐特征。
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/code:${PYTHONPATH:-}"

ROUND="${1:-}"
if [[ "$ROUND" != "A" && "$ROUND" != "B" ]]; then
    echo "用法: bash code/scripts/run_predict.sh A|B" >&2
    exit 2
fi
export FRAUD_ROUND="$ROUND"
python code/test/predict.py
