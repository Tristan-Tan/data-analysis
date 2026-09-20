# -*- coding: utf-8 -*-
"""生成一套**结构完全仿真、体量极小**的原始 CSV，供端到端自检使用。

⚠ 这不是真实数据，也**不代表任何效果验证**；它只用来验证重组后的流水线
   能否跑通、两个赛段是否真正隔离、模型能否落盘并被 predict 正确加载。

字段与《数据字典整理.md》一致；码值刻意避开 02/03/05 里硬编码的
OLD_SMY / OLD_CH 列表，以便触发"全量展开"分支。

用法: python code/selftest/make_synth_data.py <输出根目录> [规模]
"""
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260920)
# 刻意不在 02/03/05 的硬编码列表里，确保走"新增展开"分支
SMY = ["9101", "9102", "9103", "9104", "9105", "9106"]
# 只出现在 testB 的两个摘要码：它们只有在 testB 并入参考池时才够到
# 05_features_v4 的「覆盖 >=300 卡」门槛，从而多展开 2 列。
# 这正是赛段必须隔离的硬证据 —— 参考池不同，**特征集本身**就不同。
SMY_TESTB_ONLY = ["9201", "9202"]
CH = ["0201", "0202", "0203", "0204", "0205", "0206"]
CLEAR = ["on_us_bank_txn", "interbank_txn", "other_clear_txn"]


def gen(base, n_train=1500, n_testa=400, n_testb=400, n_txn_per_card=8):
    d_tr = os.path.join(base, "data", "train")
    d_a = os.path.join(base, "data", "testA")
    d_b = os.path.join(base, "data", "testB")
    for d in (d_tr, d_a, d_b):
        os.makedirs(d, exist_ok=True)

    tr = [f"TR{i:06d}" for i in range(n_train)]
    ta = [f"TA{i:06d}" for i in range(n_testa)]
    tb = [f"TB{i:06d}" for i in range(n_testb)]
    ab = tr + ta                      # A 榜可见的卡
    allc = ab + tb

    # label：2% 欺诈，且与"沉默天数长"弱相关，让 silence 族有信号可学
    y = np.zeros(n_train, dtype=int)
    y[RNG.choice(n_train, max(int(n_train * 0.02), 10), replace=False)] = 1

    # ---- txn：A/B 两批锚点相差约一个月，复刻真实的跨批次断层 ----
    rows = []
    for grp, cards, anchor in (("ab", ab, datetime(2026, 1, 1)),
                               ("b", tb, datetime(2026, 1, 31))):
        for i, c in enumerate(cards):
            lab = y[tr.index(c)] if c in tr[:n_train] and c.startswith("TR") else 0
            # 欺诈卡沉默更久（train label1 均值 16.6 天 vs label0 7.9 天）
            sil = RNG.integers(10, 25) if lab else RNG.integers(0, 10)
            last = anchor - timedelta(days=int(sil))
            n = int(RNG.integers(3, n_txn_per_card + 5))
            n_acc = 1 + int(RNG.random() < 0.15)          # 少量一卡多账号
            for j in range(n):
                t = last - timedelta(hours=int(RNG.integers(0, 24 * 25)),
                                     minutes=int(RNG.integers(0, 60)))
                is_in = RNG.random() < 0.45
                amt = float(np.round(RNG.lognormal(7, 1.2), 2))
                acc = f"AC{i:06d}{grp}{j % n_acc}"
                rows.append({
                    "card_no": c, "cst_accno": acc, "accno_txn_sn": j + 1,
                    "stm_dt": t.strftime("%Y-%m-%d"),
                    "stm_tm": t.strftime("%H:%M:%S"),
                    "tms": t.strftime("%Y-%m-%d %H:%M:%S"),
                    "acct_bal": float(np.round(RNG.lognormal(8, 1.5), 2)),
                    "txn_amt": amt,
                    "inflow_amt": amt if is_in else 0.0,
                    "outflow_amt": 0.0 if is_in else amt,
                    "cntrprt_card_no": f"CP{RNG.integers(0, 600):05d}",
                    "cntrprt_name": f"NM{RNG.integers(0, 400):05d}",
                    "cntrprt_prvt_ind": float(RNG.integers(0, 2)),
                    "cntrprt_clear_type": CLEAR[RNG.integers(0, 3)],
                    "txn_channel_type": CH[RNG.integers(0, len(CH))],
                    "txn_cgycd": "01" if RNG.random() < 0.2 else "02",
                    "smy_cd": (SMY_TESTB_ONLY[j % len(SMY_TESTB_ONLY)]
                               if grp == "b" and j < len(SMY_TESTB_ONLY)
                               else SMY[RNG.integers(0, len(SMY))]),
                    "rmrk_recharge_withdrawal_ind": float(RNG.integers(0, 2)),
                    "rmrk_short_ind": float(RNG.integers(0, 2)),
                    "rmrk_e_credit_ind": float(RNG.integers(0, 2)),
                })
    txn = pd.DataFrame(rows)
    txn[txn["card_no"].isin(ab)].to_csv(
        os.path.join(d_tr, "txn_info_train_testa.csv"), index=False)
    txn[txn["card_no"].isin(tb)].to_csv(
        os.path.join(d_b, "txn_info_testb.csv"), index=False)

    # ---- card / cst / accno ----
    def statics(cards, tag, shared_cst=0):
        """shared_cst: 让 testB 的前若干张卡复用 train 的 cst_id，
        复刻 B 榜踩过的『cst_id 跨批次重复导致行数膨胀』问题"""
        cst_ids = []
        for i, c in enumerate(cards):
            cst_ids.append(f"CS{tag}{i:06d}" if i >= shared_cst
                           else f"CSab{i:06d}")
        card = pd.DataFrame({
            "card_no": cards,
            "prim_cst_accno": [f"AC{i:06d}{tag}0" for i in range(len(cards))],
            "cst_id": cst_ids,
            "card_create_dt": [(datetime(2020, 1, 1) +
                                timedelta(days=int(RNG.integers(0, 2000))))
                               .strftime("%Y-%m-%d") for _ in cards],
            "crdisu_lvl1_insid": [f"{RNG.integers(1, 9)}00" for _ in cards],
            "crdisu_lvl2_insid": [f"{RNG.integers(1, 30)}000" for _ in cards],
            "ovsea_bsn_opn_ind": RNG.integers(0, 2, len(cards)).astype(float),
        })
        cst = pd.DataFrame({
            "cst_id": cst_ids,
            "efct_cst_ind": RNG.integers(0, 2, len(cards)).astype(float),
            "age": RNG.integers(18, 80, len(cards)),
            "gender": RNG.choice(["01", "02", "99"], len(cards)),
            "marital_status": RNG.choice(["10", "20", "40", "90"], len(cards)),
            "occupation": RNG.choice(["80000", "80200", "80400", "80500",
                                      "10700", "49900"], len(cards)),
            "industry": RNG.choice(["A", "C", "F", "O", None], len(cards)),
            "aum_val": np.round(RNG.lognormal(7, 2, len(cards)), 2),
            "mo_da_aum_val": np.round(RNG.lognormal(7, 2, len(cards)), 2),
        })
        acc = pd.DataFrame({
            "card_no": cards,
            "cst_accno": [f"AC{i:06d}{tag}0" for i in range(len(cards))],
            "open_accno_dt": [(datetime(2019, 1, 1) +
                               timedelta(days=int(RNG.integers(0, 2500))))
                              .strftime("%Y-%m-%d") for _ in cards],
        })
        return card, cst, acc

    c1, s1, a1 = statics(ab, "ab")
    # testB 前 12 张卡复用 train 的 cst_id -> 触发主键去重分支
    c2, s2, a2 = statics(tb, "b", shared_cst=12)
    c1.to_csv(os.path.join(d_tr, "card_info_train_testa.csv"), index=False)
    s1.to_csv(os.path.join(d_tr, "cst_info_train_testa.csv"), index=False)
    a1.to_csv(os.path.join(d_tr, "accno_info_train_testa.csv"), index=False)
    c2.to_csv(os.path.join(d_b, "card_info_testb.csv"), index=False)
    s2.to_csv(os.path.join(d_b, "cst_info_testb.csv"), index=False)
    a2.to_csv(os.path.join(d_b, "accno_info_testb.csv"), index=False)

    pd.DataFrame({"card_no": tr, "label": [f"{v}.0" for v in y]}).to_csv(
        os.path.join(d_tr, "train.csv"), index=False)
    pd.DataFrame({"card_no": ta}).to_csv(
        os.path.join(d_a, "testa.csv"), index=False)
    pd.DataFrame({"card_no": tb}).to_csv(
        os.path.join(d_b, "testb.csv"), index=False)

    print(f"合成数据已生成于 {base}")
    print(f"  train {len(tr)}（正例 {int(y.sum())}） testa {len(ta)} "
          f"testb {len(tb)}  txn {len(txn):,} 行")
    print(f"  testB 中有 12 张卡与 train 共用 cst_id（触发主键去重分支）")
    print(f"  摘要码 {SMY_TESTB_ONLY} 只出现在 testB："
          f"A 榜特征集里不应有 smy2_9201/9202，B 榜应有")
    return len(tr), len(ta), len(tb)


if __name__ == "__main__":
    gen(sys.argv[1] if len(sys.argv) > 1 else ".")
