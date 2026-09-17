# -*- coding: utf-8 -*-
"""pick_submissions 的合成数据自检（离线，不接触真实数据）

T1 端到端跑通
T2 top754 集合相同但 score 数值不同的两份文件，被判为"完全重复"
   （评分只看前 754 是哪些卡，数值与顺序不影响得分）
T3 机制上判死的文件（_gs_deg / r01_）被排除在选点之外
T4 已提交的文件不进入候选
T5 选点结果彼此互不相同
T6 行数或卡号集合不符的文件被跳过而不是崩溃

用法: python code/labB/selftest_pick.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
N = 3000
K = int(N * 0.01)
RNG = np.random.default_rng(3)
ok = True


def chk(name, cond, detail=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    ok = ok and bool(cond)


def main():
    base = tempfile.mkdtemp(prefix="pick_selftest_")
    try:
        it = os.path.join(base, "data", "interim")
        pr = os.path.join(base, "prediction_result")
        os.makedirs(it); os.makedirs(os.path.join(pr, "labB"))
        cards = [f"B{i:07d}" for i in range(N)]
        pd.DataFrame({"card_no": cards}).to_parquet(
            os.path.join(it, "testb.parquet"), index=False)

        def w(name, score, sub=""):
            p = os.path.join(pr, sub, name) if sub else os.path.join(pr, name)
            pd.DataFrame({"card_no": cards, "score": score}).to_csv(p, index=False)

        ref = RNG.random(N)
        w("prediction_resultB.csv", ref)
        # T2：同一个 top754 集合，但分数数值完全不同（单调变换）
        w("prediction_resultB_alias.csv", ref * 7.0 + 100)
        # 已提交的五族版
        w("prediction_resultB_fuse_plus_both_5fam.csv",
          ref + RNG.normal(0, 0.02, N))
        # 三个互不相同的未提交候选
        for i, nm in enumerate(["prediction_resultB_fuse_plus_partial.csv",
                                "prediction_resultB_moreseeds.csv",
                                "prediction_resultB_fuse_partial_x3.csv"]):
            w(nm, ref + RNG.normal(0, 0.03 * (i + 1), N))
        # 机制判死的
        w("prediction_resultB_gs_deg50_a0.05.csv", RNG.random(N))
        w("pred_testb_r01_B.csv", RNG.random(N), sub="labB")
        # 行数不符
        pd.DataFrame({"card_no": cards[:100],
                      "score": RNG.random(100)}).to_csv(
            os.path.join(pr, "prediction_resultA.csv"), index=False)

        env = dict(os.environ, FRAUD_BASE=base,
                   PYTHONPATH=os.path.join(ROOT, "code"))
        r = subprocess.run([sys.executable,
                            os.path.join(HERE, "pick_submissions.py"), "testb"],
                           env=env, capture_output=True, text=True, cwd=ROOT)
        out = r.stdout
        if r.returncode != 0:
            print(out[-2000:]); print(r.stderr[-2000:])
        print(f"\n沙盒 {base}\n" + "=" * 72)
        chk("T1 正常退出", r.returncode == 0, f"rc={r.returncode}")
        chk("T2 分数不同但 top754 相同的两份被判为完全重复",
            "prediction_resultB.csv" in out and "==" in out
            and "alias" in out)
        chk("T3 图平滑候选被排除", "gs_deg50" not in out.split("【4】")[-1])
        chk("T3 r01 候选被排除", "r01_" not in out.split("【4】")[-1])
        chk("T4 已提交文件不进入【4】选点",
            "5fam" not in out.split("【4】")[-1])
        chk("T6 行数不符的文件被跳过而非崩溃",
            "跳过 prediction_resultA.csv" in out)
        tail = out.split("【4】")[-1]
        picks = [l for l in tail.splitlines()
                 if l.strip()[:2] in ("1.", "2.", "3.")]
        chk("T5 给出了选点建议", len(picks) >= 2, f"{len(picks)} 条")
        chk("T5 选点彼此不重复",
            len({l.split()[1] for l in picks}) == len(picks))
        chk("期望值提醒仍在输出里", "期望 663" in out)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print("=" * 72)
    print("自检全部通过" if ok else "存在失败用例，请勿转入内网")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
