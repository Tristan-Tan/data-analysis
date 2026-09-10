#!/usr/bin/env bash
# 完整训练流程（含特征加工），总耗时约 2.5 小时
# 用法: bash code/scripts/run_train.sh
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/code:$PYTHONPATH"

echo "===== 01 原始 CSV -> parquet ====="            ; python code/train/01_parse_raw.py
echo "===== 02 特征 v1 (94) ====="                   ; python code/train/02_features_v1.py
echo "===== 03 特征 v2 (142) ====="                  ; python code/train/03_features_v2.py
echo "===== 04 特征 v3 (174) ====="                  ; python code/train/04_features_v3.py
echo "===== 05 特征 v4 (300) ====="                  ; python code/train/05_features_v4.py
echo "===== 06 对手边表 ====="                       ; python code/train/06_edge_tables.py
echo "===== 07 exp5 暴露编码 (fold-matched) ====="   ; python code/train/07_exp5_encoding.py testa
echo "===== 08 入金侧特征 ====="                     ; python code/train/08_inflow_feat.py
echo "===== 09 silence 特征族 ====="                 ; python code/train/09_silence_feat.py
echo "===== 10 构造 AB-thin + 训练三族模型 ====="      ; python code/train/10_train_models.py
echo "训练全部完成，模型见 model/"
