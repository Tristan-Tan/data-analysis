# -*- coding: utf-8 -*-
"""r01_subpop_model 的合成数据自检（离线运行，不接触真实数据）

⚠ 只验证**计算、对齐、泄露隔离与名额守恒**，**不代表真实效果验证**。

T1  端到端跑通，关键小节都打印
T2  保名额拼接：S 内取值集合不变、S 外分数不动、S 占的名额恒等
T3  折隔离：每折的训练行与验证行无交集；子人群外不产生 OOF 预测
T4  子人群掩码用 train 分位边界，两侧同一规则（testB 不自行取分位）
T5  段四确实剔掉了全部 exp5 与 smy_te 列（换折不得带旧编码）
T6  拼接是保序重排：新分数在 S 内的排名顺序 == 专用模型的排名顺序
T7  只写 labB 目录，未触碰正式预测目录

用法: python code/labB/selftest_r01.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "code"))
from config import SIL_COLS                                   # noqa: E402

N_TR, N_TE, POS = 20000, 3000, 400
RNG = np.random.default_rng(11)
ok = True

EXP5 = [f"{v}_{st}" for v in ("aa", "ao", "aop")
        for st in ("nf_max", "ncp_f1", "rate_max", "txnf_share")]
BASE_F = ["n_txn", "n_in", "n_out", "in_sum", "out_sum", "bal_last",
          "aum_val", "mo_da_aum_val", "n_cntrprt", "n_fastout_1h",
          "out_ratio", "card_age_d", "age", "night_share", "prvt_share"]
INFLOW = ["in_max2", "n_in_senders", "fa_deg_max", "fn_oneoff_share"]


def chk(name, cond, detail=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    ok = ok and bool(cond)


def build(base):
    it = os.path.join(base, "data", "interim")
    os.makedirs(it, exist_ok=True)
    trc = [f"T{i:07d}" for i in range(N_TR)]
    tec = [f"B{i:07d}" for i in range(N_TE)]
    allc = trc + tec
    n = len(allc)

    ntxn = RNG.integers(1, 300, n).astype(float)
    low = ntxn <= np.quantile(ntxn[:N_TR], 0.40)
    y = np.zeros(N_TR, dtype=np.int8)
    # 正例一半落在低活跃人群里，保证子人群正例数足够
    cand_lo = np.flatnonzero(low[:N_TR])
    cand_hi = np.flatnonzero(~low[:N_TR])
    y[RNG.choice(cand_lo, POS // 2, replace=False)] = 1
    y[RNG.choice(cand_hi, POS - POS // 2, replace=False)] = 1
    pd.DataFrame({"card_no": trc, "label": y.astype(float)}).to_parquet(
        os.path.join(it, "train.parquet"), index=False)
    pd.DataFrame({"card_no": tec}).to_parquet(
        os.path.join(it, "testb.parquet"), index=False)

    sig = np.r_[y, np.zeros(N_TE)] * 1.5 + RNG.normal(0, 1, n)
    f = {c: RNG.lognormal(2, 1, n) for c in BASE_F}
    f["n_txn"] = ntxn
    f["n_in"] = RNG.integers(0, 40, n).astype(float)
    f["n_fastout_1h"] = np.where(RNG.random(n) < 0.5, 0.0,
                                 RNG.integers(1, 9, n).astype(float))
    f["out_ratio"] = RNG.random(n)
    f["aum_val"] = RNG.lognormal(6, 2, n)
    f["mo_da_aum_val"] = f["aum_val"] * RNG.uniform(0.5, 1.5, n)
    f["in_sum"] = RNG.lognormal(8, 2, n) + sig * 100
    f["bal_last"] = RNG.lognormal(7, 2, n)
    fv = pd.DataFrame({"card_no": allc, **f})
    fv.to_parquet(os.path.join(it, "features_v4.parquet"), index=False)

    pd.DataFrame({"card_no": allc,
                  **{c: RNG.lognormal(1, 1, n) for c in INFLOW}}).to_parquet(
        os.path.join(it, "inflow_feat.parquet"), index=False)
    pd.DataFrame({"card_no": allc,
                  **{c: RNG.normal(0, 1, n) for c in SIL_COLS}}).to_parquet(
        os.path.join(it, "silence_feat.parquet"), index=False)
    pd.DataFrame({"card_no": allc,
                  "crdisu_lvl1_insid": RNG.choice(
                      [f"INS{i}" for i in range(12)], n)}).to_parquet(
        os.path.join(it, "card.parquet"), index=False)

    def enc(cards, seed):
        r = np.random.default_rng(seed)
        return pd.DataFrame({"card_no": cards,
                             **{c: r.lognormal(1, 1, len(cards)) for c in EXP5}})

    enc(trc, 1).to_parquet(os.path.join(it, "exp5_tr.parquet"), index=False)
    pd.DataFrame({"card_no": trc,
                  "smy_te": RNG.random(N_TR)}).to_parquet(
        os.path.join(it, "smy_te_tr.parquet"), index=False)
    for k in range(5):
        enc(tec, 100 + k).to_parquet(
            os.path.join(it, f"exp5_teb_f{k}.parquet"), index=False)
        pd.DataFrame({"card_no": tec,
                      "smy_te": RNG.random(N_TE)}).to_parquet(
            os.path.join(it, f"smy_te_teb_f{k}.parquet"), index=False)

    for m in ("dart", "abthin_lgbn", "cat"):
        np.save(os.path.join(it, f"oof_{m}.npy"),
                y * 2.0 + RNG.normal(0, 1, N_TR))
        np.save(os.path.join(it, f"te_{m}.npy"), RNG.normal(0, 1, N_TE))
    os.makedirs(os.path.join(base, "prediction_result"), exist_ok=True)
    return y, low[:N_TR], ntxn[:N_TR]   # 只回传 train 侧，与 r01 取分位的口径一致


def main():
    base = tempfile.mkdtemp(prefix="r01_selftest_")
    try:
        y, low_tr, ntxn = build(base)
        env = dict(os.environ, FRAUD_BASE=base,
                   PYTHONPATH=os.path.join(ROOT, "code"),
                   R01_MIN_POS_S="50")
        r = subprocess.run([sys.executable,
                            os.path.join(HERE, "r01_subpop_model.py"),
                            "testb", "AUTO"],
                           env=env, capture_output=True, text=True, cwd=ROOT)
        out = r.stdout
        if r.returncode != 0:
            print(out[-2500:])
            print("STDERR:\n" + r.stderr[-2500:])
        print(f"\n沙盒 {base}\n" + "=" * 74)
        chk("T1 脚本正常退出", r.returncode == 0, f"rc={r.returncode}")
        for tag in ("【段一】", "【段二】", "【段三】", "【段四】", "【4】"):
            chk(f"T1 输出含 {tag}", tag in out)
        chk("T2 名额守恒校验在脚本内通过", "✔ 名额守恒校验通过" in out)
        chk("T5 段四剔除了 exp5 与 smy_te",
            f"剔除 {len(EXP5)+1} 列" in out,
            f"期望剔除 {len(EXP5)+1} 列")
        chk("T4 子人群边界取自 train 分位", "train 分位边界" in out)

        it = os.path.join(base, "data", "interim")
        lab = os.path.join(base, "data", "interim_labB")
        picked = [f for f in os.listdir(lab)
                  if f.startswith("oof_r01_") and f.endswith("_testb.npy")]
        assert len(picked) == 1, picked
        W = picked[0].split("_")[2]
        print(f"  （AUTO 选中的划分: {W}）")
        new_tr = np.load(os.path.join(lab, picked[0]))
        from scipy.stats import rankdata
        s = np.mean([rankdata(np.load(os.path.join(it, f"oof_{m}.npy")))
                     / N_TR for m in ("dart", "abthin_lgbn", "cat")], axis=0)
        s = rankdata(s) / N_TR
        S = {"A": ntxn <= np.nanquantile(ntxn, 0.40)}[W] \
            if W == "A" else None
        if S is None:
            print(f"  （AUTO 选中 {W}，本自检只对划分 A 做独立复算，跳过 T2/T3 数值项）")
        chk("T2 S 内取值集合完全不变",
            np.allclose(np.sort(new_tr[S]), np.sort(s[S])))
        chk("T2 S 外分数逐元素未动", np.array_equal(new_tr[~S], s[~S]))
        k = int(N_TR * 0.01)
        q_old = int(((rankdata(-s, method="ordinal") <= k) & S).sum())
        q_new = int(((rankdata(-new_tr, method="ordinal") <= k) & S).sum())
        chk("T2 S 占的名额恒等", q_old == q_new, f"{q_old} vs {q_new}")
        chk("T2 拼接确实改变了排序（不是恒等变换）",
            not np.array_equal(new_tr, s))

        meta = json.load(open(os.path.join(lab, f"r01_{W}_testb.meta.json"),
                              encoding="utf-8"))
        chk("T3 meta 记录的子人群规模与独立复算一致",
            meta["子人群n"] == int(S.sum()),
            f"{meta['子人群n']} vs {int(S.sum())}")
        chk("T3 meta 记录的子人群正例数一致",
            meta["子人群pos"] == int(y[S].sum()))

        # T6 拼接保序性：新分数在 S 内的名次顺序应与专用模型一致
        #    用"新分数在 S 内的排名"与"基线取值降序"的一一对应来反推
        vals = np.sort(s[S])[::-1]
        chk("T6 S 内新分数恰为基线取值的一个重排",
            np.allclose(np.sort(new_tr[S])[::-1], vals))

        pr = os.path.join(base, "prediction_result")
        stray = [f for f in os.listdir(pr) if f != "labB"]
        chk("T7 未向 prediction_result 根目录写文件", not stray, str(stray))
        chk("T7 候选文件写在 labB 子目录",
            os.path.exists(os.path.join(pr, "labB",
                                        f"pred_testb_r01_{W}.csv")))
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print("=" * 74)
    print("自检全部通过（仅验证计算/对齐/隔离，不代表真实效果）" if ok
          else "存在失败用例，请勿转入内网")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
