# -*- coding: utf-8 -*-
"""从已训练模型导出 feature importance —— 判断绝对日历特征值不值得剔除

背景：testb_drift_check.py 得到 train vs testB 对抗 AUC = 1.0000，来源已定位：
  sil_days = 锚点 - 末笔日，而两侧锚点星期几不同（train 2026-01-01 周四，
  testB 2026-01-31 周六），因此 (weekday + sil_days) mod 7 在两侧恒为不同常数，
  构成 100% 精确的判别规则 —— AUC=1.0 是数学必然，不代表模型失效。

真正需要处理的是 anchor_weekday / S_last_wd 这两个纯绝对日历量（两侧均值
3.33 vs 4.41，系统性平移）。剔除它们要重训 2.5 小时，本脚本先零成本判断：
这两维在**真实欺诈模型**里到底有多重要。

  · 排名靠前（比如进 Top30 / gain 占比可观） -> 剔除很可能有收益，值得重训
  · 排名很靠后                               -> 解释不了 28 张落差，另找原因

用法: PYTHONPATH=code python code/explore/model_importance_check.py
耗时: 约 1 分钟（只读模型文件，不训练、不读数据）
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR, N_FOLDS, FINAL_RECIPE

# 重点关注的特征，三类：
#   ① 纯绝对日历量（跨批次不可比，考虑剔除）
#   ② 组内分位族（在 train+testA+testB 混合池里排名，两侧语义已不一致）
#   ③ 参照用的已知强特征 / 已排除嫌疑的 smy_te
CALENDAR = ["anchor_weekday", "S_last_wd", "anchor_hour", "S_last_hour",
            "last_hour"]
PCT_BIN = ["S_sil_pct_in_ntxn_bin", "S_ntxn_pct_in_sil_bin",
           "S_bal_pct_in_sil_bin", "S_insum_pct_in_sil_bin"]
WATCH = CALENDAR + PCT_BIN + ["S_sil_days", "S_ntxn_x_sil",
                              "S_ntxn_div_sil", "smy_te"]


def family_importance(files, cols):
    gain = np.zeros(len(cols))
    n = 0
    for f in files:
        p = os.path.join(MODEL_DIR, f)
        if not os.path.exists(p):
            continue
        m = lgb.Booster(model_file=p)
        g = m.feature_importance("gain")
        if len(g) != len(cols):
            print(f"  ⚠ {f}: 模型特征数 {len(g)} != 列顺序文件 {len(cols)}，跳过")
            continue
        gain += g
        n += 1
    return (gain / max(n, 1)), n


def report(name, files, cols_file):
    path = os.path.join(MODEL_DIR, cols_file)
    if not os.path.exists(path):
        print(f"\n[{name}] 缺少列顺序文件 {cols_file}，跳过")
        return
    with open(path) as f:
        cols = json.load(f)
    gain, n = family_importance(files, cols)
    if n == 0:
        print(f"\n[{name}] 没找到模型文件，跳过")
        return

    imp = pd.DataFrame({"feature": cols, "gain": gain})
    imp["share"] = imp["gain"] / imp["gain"].sum()
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    imp["rank"] = imp.index + 1

    print(f"\n{'=' * 70}")
    print(f"【{name}】{n} 个模型平均，共 {len(cols)} 维")
    print("\n-- Top20 --")
    print(imp.head(20)[["rank", "feature", "share"]].to_string(index=False))

    print("\n-- 重点关注维度 --")
    w = imp[imp["feature"].isin(WATCH)][["rank", "feature", "share"]]
    print(w.to_string(index=False) if len(w) else "  （矩阵里没有这些列）")

    cal = imp[imp["feature"].isin(CALENDAR)]
    pct = imp[imp["feature"].isin(PCT_BIN)]
    print(f"\n-- 绝对日历量 {CALENDAR} 合计 gain 占比: {cal['share'].sum():.4%}")
    print(f"-- 组内分位族 {PCT_BIN} 合计 gain 占比: {pct['share'].sum():.4%}")


report("dart（402 维）",
       [f"dart_s{sd}_f{k}.txt" for sd in FINAL_RECIPE["dart"]
        for k in range(N_FOLDS)],
       "feature_cols.json")

report("abthin_lgbn（409 维）",
       [f"abthin_lgbn_s{sd}_f{k}.txt" for sd in FINAL_RECIPE["lgbn"]
        for k in range(N_FOLDS)],
       "account_balance_thin_feature_cols.json")

print("""
【判断标准】
  · 绝对日历量合计占比 <0.5%  -> 剔除它们收益有限，但成本也几乎为零，可顺手做
  · 组内分位族占比可观（审查材料记载 sil_pct_in_ntxn_bin 是全 401 维 gain
    第 4）-> 优先修：改为按批次（train+testA / testB）分别排名，而不是在
    450800 张卡的混合池里排名，否则两侧同一分位值含义不一致
两项可在同一次重训里一起修复，只花一次 2.5 小时。""")
