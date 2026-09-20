# 数据目录说明

赛题数据为行内脱敏数据，**未随材料打包**。评审时请将原始文件放置如下：

```
data/
├── train/
│   ├── txn_info_train_testa.csv
│   ├── card_info_train_testa.csv
│   ├── cst_info_train_testa.csv
│   ├── accno_info_train_testa.csv
│   └── train.csv
├── testA/
│   └── testa.csv
├── testB/                    # 复赛的 5 个独立文件（流水不在 train_testa 里）
│   ├── testb.csv
│   ├── txn_info_testb.csv
│   ├── card_info_testb.csv
│   ├── cst_info_testb.csv
│   └── accno_info_testb.csv
├── interim_A/                # A 榜中间 parquet，脚本自动生成，无需手工放置
└── interim_B/                # B 榜中间 parquet，脚本自动生成，无需手工放置
```

## 为什么中间产物分了 A / B 两个目录

**这不是命名洁癖，是正确性要求。**

`01_parse_raw.py` 在 B 榜会把 testB 并入同一份 `txn / card / cst / accno`
（跨卡图特征必须共用参考池）。但参考池一变，两件事会同时变：

1. **特征取值**：`occupation_freq` 等 6 个频次编码、对手网络度、exp5 的低度数
   变体，全都依赖"池子里有哪些卡"；
2. **特征集本身**：`05_features_v4.py` 按「覆盖 ≥300 张卡」的门槛决定展开哪些
   摘要码与渠道码，testB 并入后更多码值够到门槛，**列数随之改变**。

A 榜成绩是在 testB 尚不存在时跑出的。两榜若共用一个 `interim/`，先跑哪个
就会污染另一个，而且**不会报任何错**。因此赛段由环境变量 `FRAUD_ROUND`
显式指定，中间产物与模型分目录存放，两榜可任意顺序、重复运行。

## 磁盘占用

单赛段中间产物约 8~10GB，两榜并存约 20GB。若空间紧张，可在 A 榜预测完成后
删除 `data/interim_A/`（保留 `model/A/` 与提交文件）；需要再次预测 A 榜时用
`bash code/scripts/run_features.sh A` 重新生成，约 25 分钟。

## 其它

- 第 10 步会从 `txn.parquet` 与 `card.parquet` 自动生成
  `interim_{A,B}/account_balance_v1.parquet`，供 AB-thin 窄树使用，无需手工准备。
- 若数据放在别处，可通过环境变量覆盖：`export FRAUD_BASE=/your/project/root`，
  或直接修改 `code/config.py` 顶部的路径常量（全部路径集中于该文件）。
