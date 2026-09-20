# code/explore/ —— 赛中诊断脚本（**不属于复现链路**）

这些脚本是 A/B 两榜期间用于验证假设、定位问题的一次性工具。它们不参与
最终成绩的复现，但构成《审查材料说明》§三（二）中"七个优化方向被系统性
排除"的证据链，故一并归档。

**运行前必须指定赛段**（这些脚本读的是赛段专属的中间产物）：

```bash
export FRAUD_ROUND=B
PYTHONPATH=code python code/explore/<脚本>.py
```

| 脚本 | 用途 |
|---|---|
| `testb_anchor_check.py` | 核对 testB 末笔日期分布，确认 silence 锚点是否需要分批 |
| `testb_drift_check.py` | 逐特征 PSI × gain 风险分，定位跨批次漂移源 |
| `domain_adapt_check.py` | 对抗验证：判断域适应在数学上是否成立 |
| `graph_smooth.py` | 团伙图平滑，用 train 真标签验证效果方向 |
| `no_silence_variant.py` | **已被 `code/train/11_train_nosil.py` 取代** |
| `fusion_candidates.py` | **已被 `code/test/predict.py` 的固化配方取代** |
| `more_seeds.py` | 多 seed 降方差 |
| `model_importance_check.py` | 特征族 gain 占比统计 |
| `field_value_check.py` / `smy_te_*.py` | 字段取值探查与 smy_te 的引入依据 |
| `post_fix_diagnosis.py` | 组内分位修复后的三族一致性诊断 |
