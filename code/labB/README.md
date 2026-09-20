# code/labB/ —— 复赛后期的独立实验目录（**不属于复现链路**）

B 榜后期为了在不触碰正式模型的前提下探索新方向，所有实验产物都写到
`data/interim_labB/` 与 `prediction_result/labB/`，与正式链路完全隔离。

这四次实验**全部被实测否定**，结论与证据见《审查材料说明》§三（二）
与 `B榜优化过程记录.md` §五。保留代码是为了让"为什么放弃"可追溯。

```bash
export FRAUD_ROUND=B
PYTHONPATH=code python code/labB/<脚本>.py testb
```

| 脚本 | 假设 | 结论 |
|---|---|---|
| `d01_error_anatomy.py` | 先测量旧模型错在哪，再决定方向 | 定位到 190 张三族共同漏报的一致画像 |
| `f01_joint_events.py` + `run_exp1.py` | 方向×摘要×渠道的联合事件 | 单模型 OOF −5，否定 |
| `f02_fund_path.py` + `run_exp2.py` | 账户内 FIFO 资金路径 | 单模型 OOF −13，否定 |
| `run_exp3.py` | exp5 换一种表达（去绝对计数） | 单模型 OOF −7，否定 |
| `r01_subpop_model.py` | 低活跃人群专用模型改善组内排序 | 组内 −24、机构留出 0/5，否定 |
| `combine_exps.py` | 三次实验的组合 | 零成本重组评估 |
| `pick_submissions.py` | 候选提交文件查重与选点 | 避免把机会浪费在重复文件上 |
| `selftest_*.py` | 各实验的合成数据自检 | 仅验证计算与对齐 |
