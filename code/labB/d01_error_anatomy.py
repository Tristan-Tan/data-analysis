# -*- coding: utf-8 -*-
"""d01 误差解剖：旧模型在哪些人群上错，分群重标定的天花板有多高

【本脚本不训练、不提交、不写预测文件】只读已有产物，输出聚合诊断。
所有输出均为分组统计量，不含任何卡号或原始流水。

回答五个问题：
  §1 旧模型的高分误报(FP)与漏报(FN)集中在哪些可解释人群
  §2 testB 头部 754 的人群构成，与 train 头部 3000 的构成差在哪
  §3 三族共同漏报的正例，是否呈现一种尚未被表示的行为
  §4 分群重标定的**天花板**（oracle，用真标签配额，不可实现，只作上界）
  §5 occupation / industry 等被丢弃的码值，有没有条件增量

前置: 10_train_models.py testb 已跑完（oof_*.npy 与 te_*.npy 齐备）
用法: PYTHONPATH=code python code/labB/d01_error_anatomy.py testb
耗时: 约 3~6 分钟，峰值内存约 3GB
"""
import glob
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P, PRED_DIR, FOLD_SEED, N_FOLDS, TOP_FRAC

t0 = time.time()
TARGET = sys.argv[1] if len(sys.argv) > 1 else "testb"
BASE3 = ["dart", "abthin_lgbn", "cat"]
MIN_POS = 20        # 正例数低于此值的分组不下结论，只报数
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)


def R(v):
    return rankdata(v) / len(v)


def sec():
    return f"{time.time()-t0:.0f}s"


# ======================= §0 现场核对 =======================
print("=" * 96)
print(f"§0 现场核对   目标集={TARGET}")
print("=" * 96)


def stat_line(path):
    st = os.stat(path)
    return (f"{os.path.basename(path):<34}{st.st_size/1e6:9.2f}MB  "
            f"{datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M}")


oof, tev = {}, {}
for m in BASE3:
    for pre, d in (("oof", oof), ("te", tev)):
        p = P(f"{pre}_{m}.npy")
        assert os.path.exists(p), f"缺少 {p}，请先跑 10_train_models.py {TARGET}"
        print("  " + stat_line(p))
        d[m] = np.load(p)

tr = pd.read_parquet(P("train.parquet"))
tr["card_no"] = tr["card_no"].astype(str)
y = tr["label"].astype(float).astype(np.int8).values
te = pd.read_parquet(P(f"{TARGET}.parquet"))
te["card_no"] = te["card_no"].astype(str)
K_TR, K_TE = int(len(y) * TOP_FRAC), int(len(te) * TOP_FRAC)
print(f"  train {len(y):,} 张 / 正例 {int(y.sum()):,} / top1%={K_TR}")
print(f"  {TARGET} {len(te):,} 张 / top1%={K_TE}")
for m in BASE3:
    assert len(oof[m]) == len(y) and len(tev[m]) == len(te), f"{m} 长度不符"

s_tr = R(np.mean([R(oof[m]) for m in BASE3], axis=0))
s_te = R(np.mean([R(tev[m]) for m in BASE3], axis=0))

sil = pd.read_parquet(P("silence_feat.parquet"),
                      columns=["card_no", "sil_days"])
sil["card_no"] = sil["card_no"].astype(str)
sil_tr = tr[["card_no"]].merge(sil, on="card_no", how="left")["sil_days"].values
sil_te = te[["card_no"]].merge(sil, on="card_no", how="left")["sil_days"].values
hard = sil_te >= 32
if hard.any():
    s_te = s_te.copy()
    s_te[hard] = 1.0
print(f"  硬规则 sil_days>=32：{TARGET} 命中 {int(hard.sum())} 张")

rk_tr = rankdata(-s_tr, method="ordinal")
rk_te = rankdata(-s_te, method="ordinal")
sel_tr, sel_te = rk_tr <= K_TR, rk_te <= K_TE
tp_base = int(y[sel_tr].sum())
print(f"  三族融合 OOF top1% 命中 {tp_base}/{K_TR}"
      f"（记录值 2772，偏差应为 0）")

folds = list(StratifiedKFold(N_FOLDS, shuffle=True,
                             random_state=FOLD_SEED).split(np.zeros(len(y)), y))
fh = []
for _, vi in folds:
    kk = int(len(vi) * TOP_FRAC)
    fh.append(int(y[vi][np.argsort(-s_tr[vi])[:kk]].sum()))
print(f"  各折 top1% 命中 {fh}  合计 {sum(fh)}")

# 与内网现存提交文件比对，确认在诊断"真正提交过的那一版"
print("\n  与 prediction_result/ 下现存 csv 的 top754 重叠：")
mine = set(te.loc[sel_te, "card_no"])
found = False
for f in sorted(glob.glob(os.path.join(PRED_DIR, "*.csv"))):
    try:
        c = pd.read_csv(f, dtype={"card_no": str})
        if "score" not in c.columns or len(c) != len(te):
            continue
        other = set(c.nlargest(K_TE, "score")["card_no"].astype(str))
        print(f"    {os.path.basename(f):<46} 重叠 {len(mine & other)}/{K_TE}")
        found = True
    except Exception as e:                                   # noqa: BLE001
        print(f"    {os.path.basename(f):<46} 读取失败 {type(e).__name__}")
if not found:
    print("    （未找到可比对的 csv；后续诊断以三族融合重建版为准）")

# ======================= 分组变量构造 =======================
print("\n" + "=" * 96)
print(f"§分组变量构造   {sec()}")
print("=" * 96)

WANT = ["n_txn", "n_in", "n_out", "in_sum", "out_sum", "n_cntrprt",
        "card_age_d", "age", "aum_val", "mo_da_aum_val", "bal_last",
        "n_accounts", "cst_n_cards", "prvt_share", "interbank_share",
        "cash_share", "night_share", "out_ratio", "n_fastout_1h",
        "fio_median", "efct_cst_ind", "n_smy", "n_channel", "span_days"]
import pyarrow.parquet as pq                            # noqa: E402
# 只读 schema，避免把 300 列全表载入内存
schema = pq.ParquetFile(P("features_v4.parquet")).schema_arrow.names
FREQ = [f"{c}_freq" for c in ("occupation", "industry", "crdisu_lvl1_insid",
                              "crdisu_lvl2_insid", "marital_status", "gender")]
use = ["card_no"] + [c for c in WANT + FREQ if c in schema]
miss = [c for c in WANT if c not in schema]
miss_freq = [c for c in FREQ if c not in schema]
if miss_freq:
    print(f"  ⚠ 频次编码列缺失（§5 的相关性对比将跳过）: {miss_freq}")
if miss:
    print(f"  ⚠ features_v4 中不存在的列（将跳过）: {miss}")
fv = pd.read_parquet(P("features_v4.parquet"), columns=use)
fv["card_no"] = fv["card_no"].astype(str)

card = pd.read_parquet(P("card.parquet"),
                       columns=["card_no", "cst_id", "crdisu_lvl1_insid"])
cst = pd.read_parquet(P("cst.parquet"),
                      columns=["cst_id", "occupation", "industry",
                               "gender", "marital_status"])
for d, cs in ((card, ["card_no", "cst_id", "crdisu_lvl1_insid"]),
              (cst, ["cst_id", "occupation", "industry",
                     "gender", "marital_status"])):
    for c in cs:
        d[c] = d[c].astype(str)
raw = card.merge(cst, on="cst_id", how="left").drop(columns=["cst_id"])
assert raw["card_no"].is_unique, "card.parquet 的 card_no 不唯一"


def frame(keys):
    d = keys[["card_no"]].merge(fv, on="card_no", how="left") \
                         .merge(raw, on="card_no", how="left")
    assert len(d) == len(keys), "分组表拼接后行数变化"
    return d


F_tr, F_te = frame(tr), frame(te)
F_tr["sil_days"], F_te["sil_days"] = sil_tr, sil_te


def qbins(col, q=5):
    """在 train 上取分位边界，两侧用同一组边界；缺失单独成组"""
    v = F_tr[col].replace([np.inf, -np.inf], np.nan)
    edges = np.unique(np.nanquantile(v, np.linspace(0, 1, q + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    def cut(f):
        x = f[col].replace([np.inf, -np.inf], np.nan)
        lab = pd.cut(x, edges, labels=False, include_lowest=True)
        s = pd.Series(np.where(pd.isna(lab), "缺失",
                               [f"Q{int(i)+1}" if not pd.isna(i) else "缺失"
                                for i in lab]), index=f.index)
        return s
    return cut(F_tr), cut(F_te)


def fixbins(col, edges, names):
    def cut(f):
        x = pd.to_numeric(f[col], errors="coerce")
        lab = pd.cut(x, edges, labels=names, right=False)
        return pd.Series(lab.astype(object)).fillna("缺失")
    return cut(F_tr), cut(F_te)


def topcodes(col, k=12):
    """按 train 内频次取 Top k 码值，其余归 '其他'，空值归 '缺失'"""
    vc = F_tr[col].replace({"None": np.nan, "nan": np.nan, "": np.nan})
    top = vc.value_counts().head(k).index.tolist()

    def cut(f):
        x = f[col].replace({"None": np.nan, "nan": np.nan, "": np.nan})
        return pd.Series(np.where(x.isna(), "缺失",
                                  np.where(x.isin(top), x, "其他")),
                         index=f.index)
    return cut(F_tr), cut(F_te)


def role(f):
    """收付角色：按资金进出结构分，语义互斥"""
    i, o = f.get("in_sum"), f.get("out_sum")
    ni, no = f.get("n_in"), f.get("n_out")
    if i is None or o is None:
        return pd.Series("未知", index=f.index)
    i, o = i.fillna(0), o.fillna(0)
    ni = ni.fillna(0) if ni is not None else pd.Series(0, index=f.index)
    no = no.fillna(0) if no is not None else pd.Series(0, index=f.index)
    ratio = o / (i + 1)
    r = pd.Series("其他", index=f.index, dtype=object)
    r[(no == 0) & (ni > 0)] = "纯收款"
    r[(ni == 0) & (no > 0)] = "纯付款"
    m = (ni > 0) & (no > 0)
    r[m & (ratio >= 0.9) & (ratio <= 1.1)] = "过路(出≈入)"
    r[m & (ratio < 0.5)] = "囤积(出<半入)"
    r[m & (ratio > 1.1)] = "净流出(出>入)"
    return r


GROUPS = {}
for name, col in [("活跃度 n_txn", "n_txn"), ("财富 月日均AUM", "mo_da_aum_val"),
                  ("对手覆盖 n_cntrprt", "n_cntrprt"), ("入金笔数 n_in", "n_in")]:
    if col in F_tr.columns:
        GROUPS[name] = qbins(col)
if "card_age_d" in F_tr.columns:
    GROUPS["卡龄"] = fixbins("card_age_d", [-np.inf, 90, 365, 1095, 3650, np.inf],
                             ["<3月", "3月-1年", "1-3年", "3-10年", "10年+"])
if "age" in F_tr.columns:
    GROUPS["年龄"] = fixbins("age", [-np.inf, 25, 35, 45, 60, np.inf],
                             ["<25", "25-35", "35-45", "45-60", "60+"])
GROUPS["沉默天数(对照)"] = fixbins("sil_days", [-np.inf, 1, 3, 8, 15, 30, np.inf],
                                   ["0", "1-2", "3-7", "8-14", "15-29", "30+"])
GROUPS["收付角色"] = (role(F_tr), role(F_te))
for name, col in [("发卡一级机构", "crdisu_lvl1_insid"), ("职业 occupation", "occupation"),
                  ("行业 industry", "industry"), ("性别", "gender"),
                  ("婚姻", "marital_status")]:
    if col in F_tr.columns:
        GROUPS[name] = topcodes(col, 12 if col != "gender" else 6)
print(f"  共构造 {len(GROUPS)} 组分组变量   {sec()}")

# ======================= §1 误差解剖 =======================
print("\n" + "=" * 96)
print("§1 误差解剖：FP(高分误报) 与 FN(漏报) 集中在哪些人群")
print("  列义：n=样本量 pos=正例数 sel=被选入top1% tp=选中且真 fp=选中但假")
print("       漏报率=fn/pos  候选区=正例中排名落在[K,2K)的数量")
print("       配额比=sel/pos（>1 该人群被过度选中，<1 被系统性低估）")
print("=" * 96)


def anatomy(g_tr):
    rows = []
    for gv, idx in pd.Series(g_tr).groupby(pd.Series(g_tr)).groups.items():
        i = np.asarray(idx)
        n, pos = len(i), int(y[i].sum())
        s = int(sel_tr[i].sum())
        tp = int(y[i][sel_tr[i]].sum())
        fn = pos - tp
        cand = int(((rk_tr[i] >= K_TR) & (rk_tr[i] < 2 * K_TR) &
                    (y[i] == 1)).sum())
        auc = np.nan
        if pos >= MIN_POS and pos < n:
            auc = roc_auc_score(y[i], s_tr[i])
        rows.append({"组": gv, "n": n, "pos": pos,
                     "正例率": pos / n, "sel": s, "tp": tp, "fp": s - tp,
                     "fn": fn, "漏报率": fn / pos if pos else np.nan,
                     "候选区": cand,
                     "配额比": s / pos if pos else np.nan,
                     "组内AUC": auc})
    d = pd.DataFrame(rows).sort_values("n", ascending=False)
    return d


ANA = {}
for name, (g_tr, g_te) in GROUPS.items():
    d = anatomy(g_tr)
    ANA[name] = d
    print(f"\n-- {name} --")
    print(d.to_string(index=False,
                      formatters={"正例率": "{:.3%}".format,
                                  "漏报率": lambda v: "-" if pd.isna(v) else f"{v:.1%}",
                                  "配额比": lambda v: "-" if pd.isna(v) else f"{v:.2f}",
                                  "组内AUC": lambda v: "-" if pd.isna(v) else f"{v:.4f}"}))
    weak = d[(d["pos"] >= MIN_POS) & (d["配额比"] < 0.7)]
    if len(weak):
        print(f"   ⚠ 配额比<0.7（被系统性低估，pos>={MIN_POS}）: "
              + ", ".join(f"{r['组']}(pos={r['pos']},配额比{r['配额比']:.2f})"
                          for _, r in weak.iterrows()))

# ======================= §2 头部构成对比 =======================
print("\n" + "=" * 96)
print(f"§2 {TARGET} 头部 {K_TE} 的人群构成 vs train 头部 {K_TR}")
print("  倍率 = testB头部占比 / train头部占比。远离 1 表示头部人群结构变了")
print("  ⚠ 全体占比的差异是批次固有差异；真正要看的是**头部**占比的差异")
print("=" * 96)
for name, (g_tr, g_te) in GROUPS.items():
    a = pd.Series(g_tr)
    b = pd.Series(g_te)
    d = pd.DataFrame({
        "train全体": a.value_counts(normalize=True),
        "train头部": a[sel_tr].value_counts(normalize=True),
        f"{TARGET}全体": b.value_counts(normalize=True),
        f"{TARGET}头部": b[sel_te].value_counts(normalize=True),
        f"{TARGET}头部n": b[sel_te].value_counts()}).fillna(0)
    d["倍率"] = d[f"{TARGET}头部"] / d["train头部"].replace(0, np.nan)
    d = d.sort_values(f"{TARGET}头部", ascending=False)
    print(f"\n-- {name} --")
    print(d.to_string(formatters={
        "train全体": "{:.2%}".format, "train头部": "{:.2%}".format,
        f"{TARGET}全体": "{:.2%}".format, f"{TARGET}头部": "{:.2%}".format,
        f"{TARGET}头部n": "{:.0f}".format,
        "倍率": lambda v: "-" if pd.isna(v) else f"{v:.2f}"}))

# ======================= §3 共同漏报 =======================
print("\n" + "=" * 96)
print("§3 三族共同漏报的正例：是不是同一种没被表示的行为")
print("=" * 96)
miss_cnt = np.zeros(len(y), dtype=int)
for m in BASE3:
    r = rankdata(-oof[m], method="ordinal")
    miss_cnt += ((r > K_TR) & (y == 1)).astype(int)
pos_idx = np.flatnonzero(y == 1)
print("  正例被几个族漏掉的分布：")
for c in range(4):
    n = int((miss_cnt[pos_idx] == c).sum())
    print(f"    被 {c} 个族漏掉: {n:>5} 张 ({n/len(pos_idx):.1%})")

allmiss = np.flatnonzero((miss_cnt == 3) & (y == 1))
partmiss = np.flatnonzero((miss_cnt > 0) & (miss_cnt < 3) & (y == 1))
hitpos = np.flatnonzero((miss_cnt == 0) & (y == 1))
print(f"\n  共同漏报 {len(allmiss)} 张 / 部分漏报 {len(partmiss)} 张 / "
      f"全族命中 {len(hitpos)} 张")
if len(allmiss) >= MIN_POS:
    print("\n  共同漏报正例在融合分上的排名分布：")
    r = rk_tr[allmiss]
    for lo, hi in [(K_TR, 2 * K_TR), (2 * K_TR, 5 * K_TR),
                   (5 * K_TR, 20 * K_TR), (20 * K_TR, len(y) + 1)]:
        n = int(((r >= lo) & (r < hi)).sum())
        print(f"    [{lo:>6},{hi:>6}): {n:>5} 张 ({n/len(allmiss):.1%})")

    print("\n  行为画像对比（中位数；差异倍率 = 共同漏报 / 全族命中）：")
    prof = [c for c in WANT if c in F_tr.columns] + ["sil_days"]
    rows = []
    for c in prof:
        v = pd.to_numeric(F_tr[c], errors="coerce").values
        a = np.nanmedian(v[allmiss])
        b = np.nanmedian(v[hitpos])
        g = np.nanmedian(v[y == 0])
        rows.append({"特征": c, "共同漏报": a, "全族命中": b, "全体负例": g,
                     "漏报/命中": a / b if b not in (0, np.nan) else np.nan})
    d = pd.DataFrame(rows)
    d["_abs"] = (np.log(d["漏报/命中"].replace(0, np.nan)).abs())
    print(d.sort_values("_abs", ascending=False).drop(columns=["_abs"])
          .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    print("\n  共同漏报正例的人群构成（占比 vs 全族命中正例）：")
    for name, (g_tr, _) in GROUPS.items():
        a = pd.Series(g_tr)
        d = pd.DataFrame({"共同漏报": a.iloc[allmiss].value_counts(normalize=True),
                          "全族命中": a.iloc[hitpos].value_counts(normalize=True),
                          "漏报n": a.iloc[allmiss].value_counts()}).fillna(0)
        d["倍率"] = d["共同漏报"] / d["全族命中"].replace(0, np.nan)
        d = d[d["漏报n"] >= 5].sort_values("倍率", ascending=False)
        if not len(d):
            continue
        head = d.head(3)
        print(f"    {name:<18} 最富集: " + "; ".join(
            f"{i}(n={int(r['漏报n'])},倍率{r['倍率']:.2f})"
            for i, r in head.iterrows()))
else:
    print(f"  共同漏报仅 {len(allmiss)} 张，低于 {MIN_POS}，不做画像")

print("\n  候选池召回上限（train 侧，回答『能不能靠扩候选池捞回来』）：")
for frac in (0.02, 0.03, 0.05, 0.10):
    kk = int(len(y) * frac)
    print(f"    Top{frac:.0%}（{kk:>6} 张）覆盖正例 "
          f"{int(y[rk_tr <= kk].sum()):>4}/{int(y.sum())} = "
          f"{y[rk_tr <= kk].sum()/y.sum():.2%}")

# ======================= §4 分群重标定天花板 =======================
print("\n" + "=" * 96)
print("§4 分群重标定的天花板（ORACLE，使用真标签配额，不可实现）")
print("  做法：保持每个人群**组内排序不变**，把 top1% 的名额按该组真实正例数分配")
print("  含义：这是『分群阈值/分群归一』这条路线的**绝对上界**。")
print("       上界很小 => 该路线数学上就没有空间，直接放弃，不必实现")
print("=" * 96)
rows = []
for name, (g_tr, _) in GROUPS.items():
    a = pd.Series(g_tr).values
    hits = 0
    for gv in pd.unique(a):
        i = np.flatnonzero(a == gv)
        quota = int(y[i].sum())
        if quota == 0:
            continue
        hits += int(y[i][np.argsort(-s_tr[i])[:quota]].sum())
    rows.append({"分组变量": name, "oracle命中": hits,
                 "基线": tp_base, "上界增益": hits - tp_base,
                 "组数": len(pd.unique(a))})
d = pd.DataFrame(rows).sort_values("上界增益", ascending=False)
print(d.to_string(index=False))
print("\n  判读：上界增益 < 20 张 => 分群重标定在 train 上就没有空间；"
      "\n        > 100 张 => 值得做，但仍需证明该配额差异能在 testB 上被无标签地估计出来")

# ======================= §5 被丢弃码值的条件增量 =======================
print("\n" + "=" * 96)
print("§5 02_features_v1.py 中被 drop 的码值：有没有条件增量")
print("  ⚠ 目标编码用 FOLD_SEED=42 的 5 折 OOF 生成，训练侧无泄露；")
print("    但该编码**不能**被换折的新实验直接复用（须随新折重建）")
print("=" * 96)
CODE_COLS = [c for c in ["occupation", "industry", "crdisu_lvl1_insid",
                         "marital_status", "gender"] if c in F_tr.columns]
band = (rk_tr >= K_TR) & (rk_tr < 2 * K_TR)      # 决定入选与否的候选区
band2 = (rk_tr >= int(0.5 * K_TR)) & (rk_tr < 2 * K_TR)
print(f"  候选区定义: 融合排名 [{K_TR},{2*K_TR}) n={int(band.sum())} "
      f"正例={int(y[band].sum())}；"
      f"[{int(0.5*K_TR)},{2*K_TR}) n={int(band2.sum())} 正例={int(y[band2].sum())}")

for c in CODE_COLS:
    x = F_tr[c].replace({"None": np.nan, "nan": np.nan, "": np.nan}).fillna("__NA__")
    xt = F_te[c].replace({"None": np.nan, "nan": np.nan, "": np.nan}).fillna("__NA__")
    prior = y.mean()
    te_oof = np.full(len(y), prior)
    for ti, vi in folds:                       # fold-safe 目标编码
        g = pd.DataFrame({"k": x.values[ti], "y": y[ti]}).groupby("k")["y"]
        mp = ((g.sum() + 20 * prior) / (g.count() + 20))
        te_oof[vi] = pd.Series(x.values[vi]).map(mp).fillna(prior).values
    auc_all = roc_auc_score(y, te_oof)
    auc_b = roc_auc_score(y[band], te_oof[band]) \
        if 0 < y[band].sum() < band.sum() else np.nan
    auc_b2 = roc_auc_score(y[band2], te_oof[band2]) \
        if 0 < y[band2].sum() < band2.sum() else np.nan
    freq_col = f"{c}_freq"
    corr_freq = np.nan
    if freq_col in F_tr.columns:
        v = pd.to_numeric(F_tr[freq_col], errors="coerce")
        corr_freq = pd.Series(te_oof).corr(v, method="spearman")
    corr_score = pd.Series(te_oof).corr(pd.Series(s_tr), method="spearman")
    cov_te = float((xt != "__NA__").mean())
    unseen = float((~xt.isin(set(x.unique()))).mean())
    print(f"\n  -- {c} --  train 取值数={x.nunique()}  "
          f"{TARGET}非空率={cov_te:.2%}  {TARGET}中未见过的码占比={unseen:.2%}")
    print(f"     单变量AUC 全局={auc_all:.4f}  "
          f"候选区[{K_TR},{2*K_TR})={auc_b if np.isnan(auc_b) else round(auc_b,4)}  "
          f"候选区[{int(.5*K_TR)},{2*K_TR})="
          f"{auc_b2 if np.isnan(auc_b2) else round(auc_b2,4)}")
    print(f"     与现有 {freq_col} 的 Spearman="
          f"{'-' if pd.isna(corr_freq) else f'{corr_freq:.4f}'}"
          f"   与融合分的 Spearman={corr_score:.4f}")
    dd = pd.DataFrame({"k": x.values, "y": y})
    g = dd.groupby("k")["y"].agg(["size", "sum"])
    g = g[g["size"] >= 200].copy()
    g["欺诈率"] = g["sum"] / g["size"]
    g["lift"] = g["欺诈率"] / prior
    g["train占比"] = g["size"] / len(y)
    tec = xt.value_counts(normalize=True)
    g[f"{TARGET}占比"] = g.index.map(tec).fillna(0)
    g["占比倍率"] = g[f"{TARGET}占比"] / g["train占比"]
    g = g.sort_values("lift", ascending=False)
    print(f"     按 lift 排序的前 8 个码（n>=200）：")
    print(g.head(8).to_string(
        formatters={"欺诈率": "{:.2%}".format, "lift": "{:.2f}".format,
                    "train占比": "{:.2%}".format,
                    f"{TARGET}占比": "{:.2%}".format,
                    "占比倍率": "{:.2f}".format}))
    if len(g) > 12:
        print(f"     按 lift 排序的后 4 个码：")
        print(g.tail(4).to_string(
            formatters={"欺诈率": "{:.2%}".format, "lift": "{:.2f}".format,
                        "train占比": "{:.2%}".format,
                        f"{TARGET}占比": "{:.2%}".format,
                        "占比倍率": "{:.2f}".format}))

print(f"\n  判读：决定性的是**候选区 AUC**。全局 AUC 高但候选区 AUC≈0.5，"
      f"\n        说明该信息只能区分头部易分样本，已被模型学走，不是增量。")

print(f"\n全部完成 {sec()}")
