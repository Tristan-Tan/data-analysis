# -*- coding: utf-8 -*-
"""f02 资金路径特征的合成数据自检（可离线运行）

覆盖五个必须正确的边界行为：
  T1 末段观察不足：观察时长 < 时间窗的入金**不计入分母**（否则系统性低估）
  T2 充分观察下的覆盖率计算正确
  T3 FIFO 匹配**不跨账户**
  T4 FIFO 部分覆盖：出金额不足时，后面的入金判为未覆盖
  T5 期初余额：入金**之前**的出金不得被错配为覆盖（负延迟必须剔除）

用法: PYTHONPATH=code python code/labB/selftest_f02.py
"""
import os
import sys
import tempfile
from datetime import datetime, timedelta

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import f02_fund_path as f02

T0 = datetime(2026, 1, 1)


def row(card, acc, sn, sec, inflow, outflow, cp):
    return {"card_no": card, "cst_accno": acc, "accno_txn_sn": sn,
            "tms": T0 + timedelta(seconds=sec),
            "inflow_amt": float(inflow), "outflow_amt": float(outflow),
            "cntrprt_card_no": cp}


ROWS = [
    # C1 观察不足：末笔就在入金后 60 秒，任何时间窗都不该把它算进分母
    row("C1", "A1", 1, 0, 100, 0, "P1"),
    row("C1", "A1", 2, 60, 0, 100, "P2"),
    # C2 充分观察：入金后 60 秒被转走，窗口一直延续到 90000 秒
    row("C2", "A2", 1, 0, 100, 0, "P1"),
    row("C2", "A2", 2, 60, 0, 100, "P2"),
    row("C2", "A2", 3, 90000, 1, 0, "P1"),
    # C3 不跨账户：A3 的入金不能被 A4 的出金覆盖
    row("C3", "A3", 1, 0, 100, 0, "P1"),
    row("C3", "A4", 1, 60, 0, 100, "P2"),
    # C4 FIFO 部分覆盖：出金 150 只够覆盖第一笔入金 100
    row("C4", "A5", 1, 0, 100, 0, "P1"),
    row("C4", "A5", 2, 10, 200, 0, "P1"),
    row("C4", "A5", 3, 20, 0, 150, "P2"),
    # C5 期初余额：出金发生在入金之前，不得判为覆盖
    row("C5", "A6", 1, 0, 0, 50, "P2"),
    row("C5", "A6", 2, 10, 30, 0, "P1"),
]


def main():
    tmp = os.path.join(tempfile.mkdtemp(), "synth_txn.parquet")
    pl.DataFrame(ROWS).write_parquet(tmp)
    print(f"合成数据 {len(ROWS)} 行 -> {tmp}\n")
    d = f02.build(txn_path=tmp).set_index("card_no")
    print("\n构造完成，开始断言\n" + "=" * 62)
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              + (f"  {detail}" if detail else ""))
        ok = ok and bool(cond)

    # T1 观察不足
    chk("T1 C1 观察不足 -> F_nobs_5m == 0",
        d.loc["C1", "F_nobs_5m"] == 0, f"实际={d.loc['C1', 'F_nobs_5m']}")
    chk("T1 C1 观察不足 -> F_cov_5m 为 NaN（而非 0）",
        np.isnan(d.loc["C1", "F_cov_5m"]), f"实际={d.loc['C1', 'F_cov_5m']}")
    chk("T1 C1 入金确实被覆盖 -> F_uncov_share == 0",
        np.isclose(d.loc["C1", "F_uncov_share"], 0.0),
        f"实际={d.loc['C1', 'F_uncov_share']}")

    # T2 充分观察
    chk("T2 C2 F_nobs_5m == 1（仅第一笔入金观察充分）",
        d.loc["C2", "F_nobs_5m"] == 1, f"实际={d.loc['C2', 'F_nobs_5m']}")
    chk("T2 C2 F_cov_5m == 1.0（60 秒内转走）",
        np.isclose(d.loc["C2", "F_cov_5m"], 1.0),
        f"实际={d.loc['C2', 'F_cov_5m']}")
    chk("T2 C2 F_cov_24h == 1.0",
        np.isclose(d.loc["C2", "F_cov_24h"], 1.0),
        f"实际={d.loc['C2', 'F_cov_24h']}")
    chk("T2 C2 F_uncov_share == 1/101（末笔入金 1 元未被覆盖）",
        np.isclose(d.loc["C2", "F_uncov_share"], 1 / 101, atol=1e-6),
        f"实际={d.loc['C2', 'F_uncov_share']}")

    # T3 不跨账户
    chk("T3 C3 跨账户不匹配 -> F_uncov_share == 1.0",
        np.isclose(d.loc["C3", "F_uncov_share"], 1.0),
        f"实际={d.loc['C3', 'F_uncov_share']}")

    # T4 FIFO 部分覆盖：覆盖 100 / 总入金 300
    chk("T4 C4 F_uncov_share == 2/3",
        np.isclose(d.loc["C4", "F_uncov_share"], 2 / 3, atol=1e-6),
        f"实际={d.loc['C4', 'F_uncov_share']}")

    # T5 期初余额：出金在入金之前，不得判为覆盖
    chk("T5 C5 负延迟被剔除 -> F_uncov_share == 1.0",
        np.isclose(d.loc["C5", "F_uncov_share"], 1.0),
        f"实际={d.loc['C5', 'F_uncov_share']}")
    chk("T5 C5 F_out_beyond_in_share == (50-30)/50 == 0.4",
        np.isclose(d.loc["C5", "F_out_beyond_in_share"], 0.4, atol=1e-6),
        f"实际={d.loc['C5', 'F_out_beyond_in_share']}")

    print("=" * 62)
    print("自检全部通过" if ok else "存在失败用例，请勿转入内网")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
