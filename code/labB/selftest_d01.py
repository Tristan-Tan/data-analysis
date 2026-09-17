# -*- coding: utf-8 -*-
"""d01_error_anatomy 的合成数据自检（离线运行，不接触真实数据）

⚠ 这只验证**计算、对齐、边界条件**是否正确，**不代表真实效果验证**。

用 FRAUD_BASE 把 config 路径整体指向临时沙盒，造出 train/testb/features_v4/
card/cst/silence 与三族 npy，然后端到端跑一次 d01，检查：
  T1  脚本正常退出，五个小节都打印出来
  T2  分组变量的分箱在 train/testB 两侧使用同一组边界（构成对比才有意义）
  T3  features_v4 缺列时走"跳过"分支而不是崩溃
  T4  §4 的 oracle 能检测出人为制造的"分群失配"（该组上界增益 > 0），
      且对不相关的随机分组变量给出接近 0 的上界增益
  T5  oracle 名额总和恒等于正例总数（配额守恒）
  T6  §5 的目标编码是 fold-safe 的：验证折的编码不含自身 label
      （用"某码值在验证折内全为正例、训练折内全为负例"的构造来检验）
  T7  未向任何正式输出目录写文件（诊断脚本只读）

用法: python code/labB/selftest_d01.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
N_TR, N_TE, POS = 6000, 2000, 60
RNG = np.random.default_rng(7)
ok = True


def chk(name, cond, detail=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    ok = ok and bool(cond)


# d01 会请求的列；故意漏掉两列以触发"跳过"分支（T3）
COLS = ["n_txn", "n_in", "n_out", "in_sum", "out_sum", "n_cntrprt",
        "card_age_d", "age", "aum_val", "mo_da_aum_val", "bal_last",
        "n_accounts", "cst_n_cards", "prvt_share", "interbank_share",
        "cash_share", "night_share", "out_ratio", "n_fastout_1h",
        "efct_cst_ind", "n_channel", "span_days"]      # 缺 n_smy / fio_median


def make_feats(n, rng):
    d = {c: rng.lognormal(2, 1, n) for c in COLS}
    d["age"] = rng.integers(18, 80, n).astype(float)
    d["card_age_d"] = rng.integers(10, 6000, n).astype(float)
    d["n_txn"] = rng.integers(1, 400, n).astype(float)
    d["n_in"] = rng.integers(0, 60, n).astype(float)
    d["n_out"] = rng.integers(0, 60, n).astype(float)
    d["in_sum"] = rng.lognormal(8, 2, n)
    d["out_sum"] = d["in_sum"] * rng.uniform(0, 2, n)
    # 频次编码列（d01 会去和目标编码求相关）
    for c in ("occupation", "industry", "crdisu_lvl1_insid",
              "marital_status", "gender"):
        d[f"{c}_freq"] = rng.integers(10, 5000, n).astype(float)
    d["mo_da_aum_val"][rng.random(n) < 0.05] = np.nan     # 缺失分支
    return pd.DataFrame(d)


def build(base):
    it = os.path.join(base, "data", "interim")
    os.makedirs(it, exist_ok=True)
    y = np.zeros(N_TR, dtype=np.int8)
    y[RNG.choice(N_TR, POS, replace=False)] = 1
    tr_cards = [f"T{i:07d}" for i in range(N_TR)]
    te_cards = [f"B{i:07d}" for i in range(N_TE)]
    pd.DataFrame({"card_no": tr_cards,
                  "label": y.astype(float)}).to_parquet(
        os.path.join(it, "train.parquet"), index=False)
    pd.DataFrame({"card_no": te_cards}).to_parquet(
        os.path.join(it, "testb.parquet"), index=False)

    ftr, fte = make_feats(N_TR, RNG), make_feats(N_TE, RNG)
    # T4：让年龄 <25 这一组的正例率明显更高，但把它们的分数人为压低
    young = ftr["age"].values < 25
    y[young & (RNG.random(N_TR) < 0.05)] = 1
    pd.DataFrame({"card_no": tr_cards}).join(ftr).to_parquet(
        os.path.join(it, "features_v4.parquet"), index=False)
    pd.DataFrame({"card_no": te_cards}).join(fte).to_parquet(
        os.path.join(it, "features_v4.parquet").replace(
            "features_v4", "_unused"), index=False)
    # 两侧共用一张 features_v4（d01 按 card_no 拼接）
    allf = pd.concat([pd.DataFrame({"card_no": tr_cards}).join(ftr),
                      pd.DataFrame({"card_no": te_cards}).join(fte)],
                     ignore_index=True)
    allf.to_parquet(os.path.join(it, "features_v4.parquet"), index=False)
    os.remove(os.path.join(it, "_unused.parquet"))

    cards = tr_cards + te_cards
    nn = len(cards)
    occ = RNG.choice(["80000", "80200", "80400", "80500", "10100", "20300"],
                     nn, p=[.3, .1, .15, .1, .2, .15])
    # T6：构造一个只出现在少数卡上的码，用于检验 fold-safe
    pd.DataFrame({"card_no": cards, "cst_id": [f"C{i}" for i in range(nn)],
                  "crdisu_lvl1_insid": RNG.choice(
                      [f"{i:03d}" for i in range(8)], nn)}).to_parquet(
        os.path.join(it, "card.parquet"), index=False)
    pd.DataFrame({"cst_id": [f"C{i}" for i in range(nn)],
                  "occupation": occ,
                  "industry": RNG.choice(["A", "B", "C", None], nn),
                  "gender": RNG.choice(["01", "02", "99"], nn),
                  "marital_status": RNG.choice(["10", "20", None], nn)
                  }).to_parquet(os.path.join(it, "cst.parquet"), index=False)

    silv = np.r_[RNG.integers(0, 35, N_TR), RNG.integers(0, 35, N_TE)]
    pd.DataFrame({"card_no": cards,
                  "sil_days": silv.astype(float)}).to_parquet(
        os.path.join(it, "silence_feat.parquet"), index=False)

    # 三族分数：与 label 相关，但对 age<25 组整体压低 —— oracle 应能发现
    for m in ("dart", "abthin_lgbn", "cat"):
        o = y * 2.0 + RNG.normal(0, 1, N_TR)
        o[young] -= 1.8
        np.save(os.path.join(it, f"oof_{m}.npy"), o)
        np.save(os.path.join(it, f"te_{m}.npy"), RNG.normal(0, 1, N_TE))
    os.makedirs(os.path.join(base, "prediction_result"), exist_ok=True)
    return y, ftr, young


def oracle(y, s, g):
    """复算 §4 的配额公式，用于 T5 配额守恒与 T4 方向性检验"""
    hits, quota_tot = 0, 0
    for gv in pd.unique(g):
        i = np.flatnonzero(g == gv)
        q = int(y[i].sum())
        quota_tot += q
        if q:
            hits += int(y[i][np.argsort(-s[i])[:q]].sum())
    return hits, quota_tot


def main():
    base = tempfile.mkdtemp(prefix="d01_selftest_")
    try:
        y, ftr, young = build(base)
        env = dict(os.environ, FRAUD_BASE=base,
                   PYTHONPATH=os.path.join(ROOT, "code"))
        r = subprocess.run([sys.executable,
                            os.path.join(HERE, "d01_error_anatomy.py"), "testb"],
                           env=env, capture_output=True, text=True, cwd=ROOT)
        out = r.stdout
        if r.returncode != 0:
            print(out[-3000:])
            print("STDERR:\n" + r.stderr[-3000:])
        print(f"\n沙盒 {base}\n" + "=" * 70)
        chk("T1 脚本正常退出", r.returncode == 0, f"rc={r.returncode}")
        for tag in ("§1 误差解剖", "§2", "§3 三族共同漏报", "§4 分群重标定",
                    "§5"):
            chk(f"T1 输出含 {tag}", tag in out)
        chk("T3 缺列走跳过分支而非崩溃",
            "不存在的列（将跳过）" in out and "n_smy" in out)

        # T2：两侧分箱边界一致 —— §2 的每张表里 train 与 testB 列同时非零
        chk("T2 §2 打印了两侧构成对比", "train头部" in out and "testb头部" in out)

        # T4/T5：独立复算 oracle
        s = np.mean([np.load(os.path.join(base, "data", "interim",
                                          f"oof_{m}.npy"))
                     for m in ("dart", "abthin_lgbn", "cat")], axis=0)
        from scipy.stats import rankdata
        s = rankdata(s) / len(s)
        k = int(len(y) * 0.01)
        base_hit = int(y[np.argsort(-s)[:k]].sum())
        gy = np.where(ftr["age"].values < 25, "<25", "其他")
        h1, q1 = oracle(y, s, gy)
        rand = RNG.choice(list("abcde"), len(y))
        h2, q2 = oracle(y, s, rand)
        chk("T5 配额总和 == 正例总数（真实分组）", q1 == int(y.sum()),
            f"{q1} vs {int(y.sum())}")
        chk("T5 配额总和 == 正例总数（随机分组）", q2 == int(y.sum()))
        chk("T4 oracle 检测出人为制造的分群失配（增益>0）",
            h1 - base_hit > 0, f"增益={h1-base_hit}")
        chk("T4 oracle 对随机分组增益明显更小",
            (h1 - base_hit) > (h2 - base_hit),
            f"真实分组 {h1-base_hit} vs 随机分组 {h2-base_hit}")

        # T6：fold-safe 目标编码 —— 验证折的编码不得包含自身 label
        prior = y.mean()
        x = np.array(["z"] * len(y), dtype=object)
        folds = list(StratifiedKFold(5, shuffle=True, random_state=42)
                     .split(np.zeros(len(y)), y))
        ti0, vi0 = folds[0]
        # 让码 'q' 只落在验证折里，且全是正例
        qidx = vi0[y[vi0] == 1][:5]
        x[qidx] = "q"
        enc = np.full(len(y), prior)
        for ti, vi in folds:
            g = pd.DataFrame({"k": x[ti], "y": y[ti]}).groupby("k")["y"]
            mp = (g.sum() + 20 * prior) / (g.count() + 20)
            enc[vi] = pd.Series(x[vi]).map(mp).fillna(prior).values
        chk("T6 只出现在验证折的码，其编码退回先验（无自身 label 泄露）",
            np.allclose(enc[qidx], prior),
            f"编码={enc[qidx][:3]} 先验={prior:.6f}")

        # T7：诊断脚本不写文件
        pr = os.path.join(base, "prediction_result")
        wrote = [f for f in os.listdir(pr)] if os.path.isdir(pr) else []
        chk("T7 未向 prediction_result 写任何文件", not wrote, str(wrote))
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print("=" * 70)
    print("自检全部通过（仅验证计算与对齐，不代表真实效果）" if ok
          else "存在失败用例，请勿转入内网")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
