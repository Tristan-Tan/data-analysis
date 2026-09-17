# -*- coding: utf-8 -*-
"""06 (卡, 对手) 带方向边表 —— 暴露编码的输入
out_n/out_sum = 该卡向该对手的出金笔数/金额；in_n/in_sum 对称；prvt 取 max
输出: data/interim/edge_acc2.parquet, edge_nm2.parquet   耗时约 2 分钟
"""
import os, sys, time
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P

t0 = time.time()
txn = pl.read_parquet(P("txn.parquet"))
base = txn.select(["card_no", "cntrprt_card_no", "cntrprt_name",
                   "cntrprt_prvt_ind", "inflow_amt", "outflow_amt"])
for keycol, name in [("cntrprt_card_no", "edge_acc2"), ("cntrprt_name", "edge_nm2")]:
    e = (base.filter(pl.col(keycol).is_not_null())
         .group_by(["card_no", keycol]).agg([
             pl.len().alias("n"),
             (pl.col("outflow_amt") > 0).sum().alias("out_n"),
             pl.col("outflow_amt").sum().alias("out_sum"),
             (pl.col("inflow_amt") > 0).sum().alias("in_n"),
             pl.col("inflow_amt").sum().alias("in_sum"),
             pl.col("cntrprt_prvt_ind").max().alias("prvt")])
         .rename({keycol: "key"}))
    e.write_parquet(P(name + ".parquet"))
    print(f"{name}: {e.shape}")
print(f"06 完成 {time.time()-t0:.0f}s")
