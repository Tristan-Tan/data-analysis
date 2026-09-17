# -*- coding: utf-8 -*-
"""combine_exps 的合成数据自检（可离线运行，不接触真实数据）

用 FRAUD_BASE 环境变量把 config 的路径整体指向一个临时沙盒，造出
train/testb/silence_feat 与 6 个成员的 oof/te npy，然后：
  A. 端到端跑通 combine_exps.py（检查有无异常、断言是否成立）
  B. 单独验证族权重口径：三个 exp 合成一族 vs 各自独立成族，
     base3 的有效权重必须分别是 3/4 与 3/6，且两种口径结果不同

用法: python code/labB/selftest_combine.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
N_TR, N_TE, POS = 3000, 2000, 60
RNG = np.random.default_rng(0)
ok = True


def chk(name, cond, detail=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    ok = ok and bool(cond)


def build_sandbox(base):
    interim = os.path.join(base, "data", "interim")
    os.makedirs(interim, exist_ok=True)
    y = np.zeros(N_TR, dtype=np.int8)
    y[RNG.choice(N_TR, POS, replace=False)] = 1
    pd.DataFrame({"card_no": [f"T{i:06d}" for i in range(N_TR)],
                  "label": y.astype(float)}).to_parquet(
        os.path.join(interim, "train.parquet"), index=False)
    cards = [f"B{i:06d}" for i in range(N_TE)]
    pd.DataFrame({"card_no": cards}).to_parquet(
        os.path.join(interim, "testb.parquet"), index=False)
    # 一部分卡触发 sil_days>=32 硬规则，另有缺失值
    sil = RNG.integers(0, 40, N_TE).astype(float)
    sil[RNG.choice(N_TE, 30, replace=False)] = np.nan
    pd.DataFrame({"card_no": cards, "sil_days": sil}).to_parquet(
        os.path.join(interim, "silence_feat.parquet"), index=False)

    # 三个基础族：各自带一部分真信号 + 独立噪声
    signal = y + RNG.normal(0, 1.0, N_TR)
    for m in ("dart", "abthin_lgbn", "cat"):
        np.save(os.path.join(interim, f"oof_{m}.npy"),
                signal + RNG.normal(0, 0.8, N_TR))
        np.save(os.path.join(interim, f"te_{m}.npy"), RNG.normal(0, 1, N_TE))
    # 三个 exp：彼此高度相关（共享同一 shared 成分），模拟同超参同折
    lab = os.path.join(base, "data", "interim_labB")
    os.makedirs(lab, exist_ok=True)
    shared = signal + RNG.normal(0, 0.5, N_TR)
    shared_te = RNG.normal(0, 1, N_TE)
    for e in ("exp1", "exp2", "exp3"):
        np.save(os.path.join(lab, f"oof_{e}_testb.npy"),
                shared + RNG.normal(0, 0.15, N_TR))
        np.save(os.path.join(lab, f"te_{e}_testb.npy"),
                shared_te + RNG.normal(0, 0.15, N_TE))
    return y


def weight_check():
    """族权重口径：rank 平均下，base3 的有效权重 3/4 vs 3/6"""
    print("\nB. 族权重口径验证（纯算术，不依赖数据）")
    R = lambda v: rankdata(v) / len(v)
    n = 500
    b = {m: RNG.normal(0, 1, n) for m in ("dart", "abthin_lgbn", "cat")}
    e = {k: RNG.normal(0, 1, n) for k in ("exp1", "exp2", "exp3")}

    def blend(groups):
        fam = [np.mean([R(v) for v in g], axis=0) for g in groups]
        return R(np.mean([R(f) for f in fam], axis=0))

    grouped = blend([[b[m]] for m in b] + [[e[k] for k in e]])
    separate = blend([[b[m]] for m in b] + [[e[k]] for k in e])
    chk("两种口径给出不同排序（说明口径选择确实影响结果）",
        not np.allclose(grouped, separate),
        f"最大排名差={np.abs(rankdata(-grouped) - rankdata(-separate)).max():.0f}")
    # 若三个 exp 完全相同，合成一族与独立成族的差别 == 权重差别
    same = {k: e["exp1"] for k in e}
    g2 = blend([[b[m]] for m in b] + [[same[k] for k in same]])
    s2 = blend([[b[m]] for m in b] + [[same[k]] for k in same])
    chk("三个 exp 完全相同时，合成一族 = base3 + 该 exp 四族等权",
        np.allclose(g2, blend([[b[m]] for m in b] + [[e["exp1"]]])))
    chk("三个 exp 完全相同时，独立成族 != 四族等权（权重被稀释为 3/6）",
        not np.allclose(s2, g2))


def main():
    base = tempfile.mkdtemp(prefix="combine_selftest_")
    try:
        print(f"A. 端到端跑通（沙盒 {base}）\n")
        build_sandbox(base)
        env = dict(os.environ, FRAUD_BASE=base,
                   PYTHONPATH=os.path.join(ROOT, "code"))
        r = subprocess.run([sys.executable,
                            os.path.join(HERE, "combine_exps.py"), "testb"],
                           env=env, capture_output=True, text=True, cwd=ROOT)
        print(r.stdout[-4000:])
        if r.returncode != 0:
            print(r.stderr[-3000:])
        chk("combine_exps.py 正常退出", r.returncode == 0,
            f"returncode={r.returncode}")
        chk("打印了组合对比表", "换入正例" in r.stdout)
        chk("打印了边界条件区分能力", "边界条件区分能力" in r.stdout)
        chk("打印了新入选卡的成员归因", "新入选卡由哪些成员推上来" in r.stdout)
        cand = os.path.join(base, "prediction_result", "labB",
                            "pred_testb_combine_best.csv")
        chk("候选文件已落盘且行数 = testb 卡数", os.path.exists(cand)
            and len(pd.read_csv(cand)) == N_TE)
        if os.path.exists(cand):
            sub = pd.read_csv(cand)
            chk("候选文件只有 card_no/score 两列",
                list(sub.columns) == ["card_no", "score"])
        # 未触碰正式目录
        chk("未写入 prediction_result 根目录",
            not [f for f in os.listdir(
                os.path.join(base, "prediction_result")) if f != "labB"])
        weight_check()
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print("\n" + "=" * 62)
    print("自检全部通过" if ok else "存在失败用例，请勿转入内网")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
