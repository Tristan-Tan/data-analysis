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
├── testB/                       # 复赛阶段
│   ├── testb.csv
│   ├── txn_info_testb.csv
│   ├── card_info_testb.csv
│   ├── cst_info_testb.csv
│   └── accno_info_testb.csv
└── interim/                     # 脚本自动生成的中间 parquet，无需手工放置
```

第 10 步还会从 `txn.parquet` 与 `card.parquet` 自动生成
`interim/account_balance_v1.parquet`，供最终 408 维 AB-thin 窄树使用。

若数据放在别处，可通过环境变量覆盖：`export FRAUD_BASE=/your/project/root`，
或直接修改 `code/config.py` 顶部的路径常量。
