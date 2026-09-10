#!/usr/bin/env bash
# 仅预测（加载 model/ 下已训练模型），耗时约 3 分钟
# 用法: bash code/scripts/run_predict.sh [testa|testb]
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/code:$PYTHONPATH"
TARGET="${1:-testa}"
python code/test/predict.py "$TARGET"
