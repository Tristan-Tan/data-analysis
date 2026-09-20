#!/usr/bin/env bash
# 只跑特征加工（01~09），不训练。用于「拿到随材料提交的 model/，
# 想直接预测但缺中间产物」的场景，约 25 分钟。
# 用法: bash code/scripts/run_features.sh A|B
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/code:${PYTHONPATH:-}"
ROUND="${1:-}"
if [[ "$ROUND" != "A" && "$ROUND" != "B" ]]; then
    echo "用法: bash code/scripts/run_features.sh A|B" >&2; exit 2
fi
export FRAUD_ROUND="$ROUND"
for s in 01_parse_raw 02_features_v1 03_features_v2 04_features_v3 \
         05_features_v4 06_edge_tables 07_exp5_encoding; do
    echo; echo "===== $s ====="; python "code/train/$s.py"
done
if [[ "$ROUND" == "B" ]]; then
    echo; echo "===== 07b_smy_te_encoding ====="
    python code/train/07b_smy_te_encoding.py
fi
for s in 08_inflow_feat 09_silence_feat; do
    echo; echo "===== $s ====="; python "code/train/$s.py"
done
echo; echo "特征加工完成，可执行: bash code/scripts/run_predict.sh $ROUND"
