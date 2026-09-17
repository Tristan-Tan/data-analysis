# -*- coding: utf-8 -*-
"""B 榜数据到手后第一步必跑：核对 testB 末笔日期分布是否与 train/testA 对齐

对应审查材料 §三（一）承诺的检查。silence 特征族贡献了模型约 13.8% 的 gain，
若 testB 末笔日期分布整体平移而未察觉，sil_days 会系统性错位，成绩大幅下滑。

前置: 已用更新后的 01_parse_raw.py 重新跑过（testB 的 5 个文件已放入
      data/testB/ 且已被合并进 txn/card/cst/accno/anchor.parquet）

用法: PYTHONPATH=code python code/explore/testb_anchor_check.py

判断标准（见脚本末尾打印的结论）：
  · 若 testB 的 last_dt 分布与 train/testA 一致（只是整体日期更靠后、内部
    形态相似），09_silence_feat.py 默认取"全量末笔日期最大值"作为锚点即可，
    不需要改代码。
  · 若 testB 内部分布出现结构性异常（如 sil_beyond_norm 尾部占比和
    train/testA 差异很大），需要改用 testB 自身 last_dt 的 P99.9 分位作为
    锚点，并同步重算训练侧 sil_days（见审查材料 §三（一）方案，需要另外
    改 09_silence_feat.py，到时候找我要代码）。
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P

t0 = time.time()

anchor = pl.read_parquet(P("anchor.parquet")).to_pandas()
if "testb" not in anchor["grp"].unique():
    print("anchor.parquet 里没有 testb 分组，请先用更新后的 01_parse_raw.py 重新跑一遍")
    sys.exit(1)

anchor["grp2"] = anchor["grp"].map({"0": "train_label0", "1": "train_label1",
                                    "testa": "testa", "testb": "testb"})
anchor["last_dt"] = pd.to_datetime(anchor["last_dt"])

print("=" * 60)
print("【1】各组 last_dt 分布（min / p50 / p99 / max）")
for g, sub in anchor.groupby("grp2"):
    q = sub["last_dt"].quantile([0, 0.5, 0.99, 1.0])
    print(f"  {g:<14s} n={len(sub):<8d} "
          f"min={q[0.0].date()} p50={q[0.5].date()} "
          f"p99={q[0.99].date()} max={q[1.0].date()}")

# ---- 按当前默认口径（全量最大值）计算 sil_days ----
global_anchor = anchor["last_dt"].max()
anchor["sil_days"] = (global_anchor - anchor["last_dt"]).dt.days
print(f"\n全量末笔日期最大值（默认锚点）: {global_anchor.date()}")

print("\n" + "=" * 60)
print("【2】各组 sil_days 分布（均值 / 中位数 / p99）")
for g, sub in anchor.groupby("grp2"):
    print(f"  {g:<14s} mean={sub['sil_days'].mean():.2f} "
          f"median={sub['sil_days'].median():.1f} "
          f"p99={sub['sil_days'].quantile(0.99):.1f}")

print("\n" + "=" * 60)
print("【3】高沉默尾部占比对比（sil_days>=30，同审查材料已做过的 train/testA 自检）")
for g, sub in anchor.groupby("grp2"):
    share = (sub["sil_days"] >= 30).mean()
    print(f"  {g:<14s} sil>=30 占比: {share:.4%}")

print("\n" + "=" * 60)
print("【4】testB 是否有异常早的末笔日期（可能是数据本身就不该参与该窗口的脏数据）")
testb = anchor[anchor["grp2"] == "testb"]
train_min = anchor[anchor["grp2"].isin(["train_label0", "train_label1"])]["last_dt"].min()
early = (testb["last_dt"] < train_min).sum()
print(f"  testB 中 last_dt 早于 train 最早值({train_min.date()}) 的卡数: {early}")

print(f"\n全部完成 {time.time()-t0:.0f}s")
print("\n请把【1】【2】【3】的完整输出发回来，据此判断是否需要改用 P99.9 锚点方案。")
