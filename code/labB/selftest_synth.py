# -*- coding: utf-8 -*-
"""f01 特征构造的合成数据自检（可离线运行，不需要真实数据）

覆盖四个必须正确的边界行为：
  T1 bigram **不跨账户**：同卡不同 cst_accno 的相邻交易不得配对
  T2 长间隔断开：间隔 > GAP_BREAK_SEC 的相邻交易不成对
  T3 分母不足写 NaN 而非 0：无有效 bigram / 无 I→O 对的卡，占比列为空
  T4 末 N 笔口径与占比计算正确（rn_end 按 card_no 倒序，与现有 04/05 一致）

用法: PYTHONPATH=code python code/labB/selftest_synth.py
"""
import os
import sys
import tempfile
from datetime import datetime, timedelta

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import f01_joint_events as f01

T0 = datetime(2026, 1, 1, 0, 0, 0)


def row(card, acc, sn, sec, inflow, outflow, smy, ch):
    return {"card_no": card, "cst_accno": acc, "accno_txn_sn": sn,
            "tms": T0 + timedelta(seconds=sec),
            "inflow_amt": float(inflow), "outflow_amt": float(outflow),
            "txn_amt": float(inflow or outflow),
            "smy_cd": smy, "txn_channel_type": ch}


ROWS = [
    # C1：两个账户。A1 内 in->out 间隔 60s 应配成 IO_5m；A2 的出金与 A1 不配对
    row("C1", "A1", 1, 0, 100, 0, "S1", "CH1"),
    row("C1", "A1", 2, 60, 0, 50, "S2", "CH2"),
    row("C1", "A2", 1, 120, 0, 30, "S2", "CH1"),
    # C2：单账户但间隔 4 天 > 3 天阈值，应断开
    row("C2", "A3", 1, 0, 200, 0, "S1", "CH1"),
    row("C2", "A3", 2, 4 * 86400, 0, 100, "S2", "CH1"),
    # C3：仅一笔交易
    row("C3", "A4", 1, 0, 50, 0, "S1", "CH1"),
    # C4：5 笔，验证末 3 笔口径
    row("C4", "A5", 1, 0, 10, 0, "S1", "CH1"),
    row("C4", "A5", 2, 10, 0, 10, "S2", "CH1"),
    row("C4", "A5", 3, 20, 0, 10, "S2", "CH1"),
    row("C4", "A5", 4, 30, 10, 0, "S1", "CH1"),
    row("C4", "A5", 5, 40, 0, 10, "S2", "CH1"),
]


def main():
    f01.PARAMS.update(TOP_DIR_SMY=10, TOP_DIR_CH=10, MIN_CARDS=1, TAIL_N=3,
                      GAP_BREAK_SEC=3 * 86400)
    tmp = os.path.join(tempfile.mkdtemp(), "synth_txn.parquet")
    pl.DataFrame(ROWS).write_parquet(tmp)
    print(f"合成数据 {len(ROWS)} 行 -> {tmp}\n")

    d = f01.build(txn_path=tmp).to_pandas().set_index("card_no")
    print("\n构造完成，开始断言\n" + "=" * 60)
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
        ok = ok and bool(cond)

    # T1 不跨账户：C1 只应有 1 个有效 bigram（A1 内），A2 那笔无前序
    chk("T1 bigram 不跨账户：C1 J_n_bigram==1",
        d.loc["C1", "J_n_bigram"] == 1, f"实际={d.loc['C1', 'J_n_bigram']}")
    chk("T1 该 bigram 落在 IO_5m",
        np.isclose(d.loc["C1", "J_bg_IO_5m"], 1.0),
        f"实际={d.loc['C1', 'J_bg_IO_5m']}")
    chk("T1 其他 bigram 档为 0（有分母，非 NaN）",
        np.isclose(d.loc["C1", "J_bg_II_5m"], 0.0),
        f"实际={d.loc['C1', 'J_bg_II_5m']}")
    # C1 的 I->O 金额比 50/100=0.5 -> 落 "50_95" 档
    chk("T1 I→O 金额比例 0.5 落 50_95 档",
        np.isclose(d.loc["C1", "J_ioratio_50_95"], 1.0),
        f"实际={d.loc['C1', 'J_ioratio_50_95']}")

    # T2 长间隔断开
    chk("T2 长间隔断开：C2 J_n_bigram==0",
        d.loc["C2", "J_n_bigram"] == 0, f"实际={d.loc['C2', 'J_n_bigram']}")
    chk("T2 C2 的 bigram 占比为 NaN",
        np.isnan(d.loc["C2", "J_bg_IO_1d"]),
        f"实际={d.loc['C2', 'J_bg_IO_1d']}")

    # T3 单笔卡
    chk("T3 单笔卡 C3 J_n_bigram==0",
        d.loc["C3", "J_n_bigram"] == 0)
    chk("T3 单笔卡 C3 bigram 占比为 NaN",
        np.isnan(d.loc["C3", "J_bg_IO_5m"]))
    chk("T3 单笔卡 C3 J_n_io==0 且比例列为 NaN",
        d.loc["C3", "J_n_io"] == 0 and np.isnan(d.loc["C3", "J_ioratio_lt50"]))

    # T4 末 N 笔：C4 全窗口 I_S1=2/5、O_S2=3/5；末 3 笔为 (O_S2, I_S1, O_S2)
    chk("T4 C4 全窗口 I_S1 占比 = 2/5",
        np.isclose(d.loc["C4", "J_ds_I_S1"], 0.4),
        f"实际={d.loc['C4', 'J_ds_I_S1']}")
    chk("T4 C4 全窗口 O_S2 占比 = 3/5",
        np.isclose(d.loc["C4", "J_ds_O_S2"], 0.6),
        f"实际={d.loc['C4', 'J_ds_O_S2']}")
    chk("T4 C4 末3笔 J_tail_n == 3",
        d.loc["C4", "J_tail_n"] == 3, f"实际={d.loc['C4', 'J_tail_n']}")
    chk("T4 C4 末3笔 O_S2 占比 = 2/3",
        np.isclose(d.loc["C4", "J_dst_O_S2"], 2 / 3),
        f"实际={d.loc['C4', 'J_dst_O_S2']}")
    chk("T4 C4 末3笔 I_S1 占比 = 1/3",
        np.isclose(d.loc["C4", "J_dst_I_S1"], 1 / 3),
        f"实际={d.loc['C4', 'J_dst_I_S1']}")

    # C4 账户内 4 个 bigram，间隔均 10s：II/IO/OI/OO 分布
    # 序列 I,O,O,I,O -> bigram: IO, OO, OI, IO  => IO=2/4, OO=1/4, OI=1/4
    chk("T4 C4 bigram IO 占比 = 2/4",
        np.isclose(d.loc["C4", "J_bg_IO_5m"], 0.5),
        f"实际={d.loc['C4', 'J_bg_IO_5m']}")
    chk("T4 C4 bigram OO 占比 = 1/4",
        np.isclose(d.loc["C4", "J_bg_OO_5m"], 0.25))
    chk("T4 C4 bigram OI 占比 = 1/4",
        np.isclose(d.loc["C4", "J_bg_OI_5m"], 0.25))

    print("=" * 60)
    print("自检全部通过" if ok else "存在失败用例，请勿转入内网")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
