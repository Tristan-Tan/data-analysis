# -*- coding: utf-8 -*-
"""r01 低活跃人群专用模型：只改组内排序，不改组间名额

【假设】d01 §3 显示 190 张三族共同漏报正例画像高度一致：高净值(aum 中位数
8488 vs 命中正例 209)、低频(n_txn 6.5 vs 38)、纯收款(倍率 4.74)、
无快速转出(n_fastout_1h 中位数 0 vs 5)、短沉默(6 天 vs 18 天)，且 88.9%
排在 [3000,6000)。模型的三大支柱——快进快出族、silence 族(13.8% gain)、
财富归一 5 维——对这批卡**全部给出反向信号**。
原因推测：GBDT 分裂阈值由多数正例主导（活跃度 Q5 有 1310 正例，Q1 只有
334），Q1 人群内部的判别维度被 Q5 的模式压掉。

【新方法】按 label-free 规则切出低活跃人群 S，在 S 内部单独训练，分裂阈值
只对这群人校准；再用**保名额拼接**装回全局。

【保名额拼接】把基线分数在 S 内的**取值集合**按专用模型的顺序重新分配给 S
内的卡。于是 sorted(new[S]) == sorted(base[S])、new[~S] == base[~S]，
S 在全局 top1% 里占的名额**恰好不变**，增益 100% 来自组内排序改善。
代码内有断言强制保证这一点。

【⚠ 折划分不可改】exp5 编码在 FOLD_SEED=42 下 fold-safe 生成。子人群模型
**沿用原折**，只把每折的训练集与验证集**都**交上子人群掩码；若重新分折，
验证折样本会落进旧编码的训练折，构成泄露。段四的机构留出验证因此**剔除
全部 exp5 与 smy_te 列**，那里才允许自由换折。

【判定】组内 top-k 命中不高于基线 / 全局换入正例 <= 换出正例 /
逐折方向不一致 / 机构留出方向反转 —— 任一成立则该路线作废。

前置: 10_train_models.py testb 已跑完
用法: PYTHONPATH=code python code/labB/r01_subpop_model.py testb [AUTO|A|B|C]
     不传第二个参数=AUTO：在正例数达标的划分里自动选组内上界最大的那个
耗时: 约 12~20 分钟
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (P, FOLD_SEED, N_FOLDS, LGB_BASE, LGBN_EXTRA, TOP_FRAC,
                    top1_f1)
from lab_common import LP, LAB_PRED, R
import assemble

t0 = time.time()
TARGET = sys.argv[1] if len(sys.argv) > 1 else "testb"
WHICH = (sys.argv[2] if len(sys.argv) > 2 else "AUTO").upper()
BASE3 = ["dart", "abthin_lgbn", "cat"]
EXP5_SUFFIX = ("_nf_max", "_ncp_f1", "_ncp_f2", "_amt_f",
               "_rate_max", "_rate_wavg", "_txnf_share")
print(f"目标集={TARGET}   子人群划分={WHICH}")


def sec():
    return f"{time.time()-t0:.0f}s"


# ==================== 段零：基线重建 ====================
print("\n" + "=" * 90)
print("【段零】基线重建与对齐核对")
print("=" * 90)
tr, Xtr, y, cols = assemble.build_train()
te, Xte_list = assemble.build_test_folds(TARGET, N_FOLDS, cols)
print(f"  训练矩阵 {Xtr.shape} / {TARGET} {Xte_list[0].shape}   {sec()}")

oof = {m: np.load(P(f"oof_{m}.npy")) for m in BASE3}
tev = {m: np.load(P(f"te_{m}.npy")) for m in BASE3}
s_tr = R(np.mean([R(oof[m]) for m in BASE3], axis=0))
s_te = R(np.mean([R(tev[m]) for m in BASE3], axis=0))

sil = pd.read_parquet(P("silence_feat.parquet"),
                      columns=["card_no", "sil_days"])
sil["card_no"] = sil["card_no"].astype(str)
sv_te = te[["card_no"]].merge(sil, on="card_no", how="left")["sil_days"].values
hard = sv_te >= 32
if hard.any():
    s_te = s_te.copy()
    s_te[hard] = 1.0

K_TR, K_TE = int(len(y) * TOP_FRAC), int(len(te) * TOP_FRAC)
rk_tr = rankdata(-s_tr, method="ordinal")
rk_te = rankdata(-s_te, method="ordinal")
sel_tr, sel_te = rk_tr <= K_TR, rk_te <= K_TE
tp_base = int(y[sel_tr].sum())
print(f"  基线三族融合 OOF top1% 命中 {tp_base}/{K_TR}（d01 实测 2772）")
print(f"  全局 FP 总数 {K_TR - tp_base} —— 这是『组内排序』路线的全局绝对上界")

folds = list(StratifiedKFold(N_FOLDS, shuffle=True,
                             random_state=FOLD_SEED).split(Xtr, y))

# ==================== 段一：子人群划分 ====================
print("\n" + "=" * 90)
print("【段一】子人群候选划分（全部 label-free，边界取自 train，两侧同一规则）")
print("=" * 90)


def col(X, name):
    return pd.to_numeric(X[name], errors="coerce").values if name in X.columns \
        else None


q_ntxn40 = np.nanquantile(col(Xtr, "n_txn"), 0.40)
q_ntxn60 = np.nanquantile(col(Xtr, "n_txn"), 0.60)
print(f"  train 分位边界: n_txn q40={q_ntxn40:.0f}  q60={q_ntxn60:.0f}")


def make_masks(X):
    ntxn = col(X, "n_txn")
    nin = col(X, "n_in")
    nfo = col(X, "n_fastout_1h")
    m = {"A": ntxn <= q_ntxn40}
    m["B"] = (nin <= 3) if nin is not None else None
    m["C"] = ((nfo == 0) & (ntxn <= q_ntxn60)) \
        if (nfo is not None and ntxn is not None) else None
    return {k: (v if v is None else np.nan_to_num(v, nan=False).astype(bool))
            for k, v in m.items()}


M_tr, M_te = make_masks(Xtr), make_masks(Xte_list[0])
DESC = {"A": "n_txn <= q40（低活跃）",
        "B": "n_in <= 3（低频收款）",
        "C": "n_fastout_1h == 0 且 n_txn <= q60（无快速转出+中低活跃）"}
rows = []
for k in ("A", "B", "C"):
    if M_tr.get(k) is None:
        print(f"  ⚠ 划分 {k} 所需列缺失，跳过")
        continue
    S = M_tr[k]
    n, pos = int(S.sum()), int(y[S].sum())
    s_in = int((sel_tr & S).sum())
    tp_in = int(y[sel_tr & S].sum())
    rows.append({"划分": k, "定义": DESC[k], "train n": n, "pos": pos,
                 "正例率": f"{pos/max(n,1):.3%}",
                 "当前占名额": s_in, "其中命中": tp_in,
                 "组内FP(=上界)": s_in - tp_in,
                 "组内漏报": pos - tp_in,
                 f"{TARGET} n": int(M_te[k].sum()),
                 f"{TARGET}占名额": int((sel_te & M_te[k]).sum())})
print(pd.DataFrame(rows).to_string(index=False))
print("  『组内FP』= 该人群内组内排序做到完美时能多命中的张数，是本路线的上界")

# 正例下限保护：低于此值训不出稳定模型。仅合成自检会下调该阈值。
MIN_POS_S = int(os.environ.get("R01_MIN_POS_S", "200"))
if WHICH == "AUTO":
    ok_rows = [r for r in rows if r["pos"] >= MIN_POS_S]
    assert ok_rows, f"没有任何划分的正例数达到下限 {MIN_POS_S}"
    best = max(ok_rows, key=lambda r: r["组内FP(=上界)"])
    WHICH = best["划分"]
    print(f"\n  AUTO 选择：在 pos>={MIN_POS_S} 的划分里取组内上界最大者 "
          f"-> {WHICH}（上界 +{best['组内FP(=上界)']}）")
S_tr, S_te = M_tr[WHICH], M_te[WHICH]
assert S_tr is not None, f"划分 {WHICH} 不可用"
n_pos_S = int(y[S_tr].sum())
assert n_pos_S >= MIN_POS_S, f"子人群正例仅 {n_pos_S}，低于下限 {MIN_POS_S}"
m_quota = int((sel_tr & S_tr).sum())
print(f"\n  本次采用划分 {WHICH}: {DESC[WHICH]}")
print(f"  子人群 n={int(S_tr.sum()):,} pos={n_pos_S} 占名额 m={m_quota} "
      f"组内上界 +{m_quota - int(y[sel_tr & S_tr].sum())}")

# ==================== 段二：子人群专用模型 ====================
print("\n" + "=" * 90)
print("【段二】子人群专用模型（沿用 FOLD_SEED=42 原折，训练/验证均交子人群掩码）")
print("=" * 90)
params = dict(LGB_BASE, seed=FOLD_SEED)
params.update(LGBN_EXTRA)

spec_oof = np.full(len(y), np.nan)
spec_te = np.zeros(len(te))
imp = np.zeros(len(cols))
fold_stat = []
for k, (ti, vi) in enumerate(folds):
    ti_s = ti[S_tr[ti]]
    vi_s = vi[S_tr[vi]]
    assert len(np.intersect1d(ti_s, vi_s)) == 0, "训练/验证折重叠"
    m = lgb.train(params, lgb.Dataset(Xtr.iloc[ti_s][cols], y[ti_s]), 3000,
                  valid_sets=[lgb.Dataset(Xtr.iloc[vi_s][cols], y[vi_s])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    spec_oof[vi_s] = m.predict(Xtr.iloc[vi_s][cols],
                               num_iteration=m.best_iteration)
    sub_te = np.flatnonzero(S_te)
    spec_te[sub_te] += m.predict(Xte_list[k].iloc[sub_te][cols],
                                 num_iteration=m.best_iteration) / N_FOLDS
    imp += m.feature_importance("gain") / N_FOLDS
    # 折内：该子人群在本折占的名额，基线命中 vs 专用模型命中
    kq = int((rk_tr[vi_s] <= K_TR).sum())
    base_hit = int(y[vi_s][np.argsort(-s_tr[vi_s])[:kq]].sum()) if kq else 0
    spec_hit = int(y[vi_s][np.argsort(-spec_oof[vi_s])[:kq]].sum()) if kq else 0
    fold_stat.append((kq, base_hit, spec_hit))
    print(f"  fold{k} 训练{len(ti_s):>7,} 验证{len(vi_s):>6,} "
          f"best_iter={m.best_iteration:<5} 折内名额{kq:<4} "
          f"基线命中{base_hit:<4} 专用命中{spec_hit:<4} "
          f"({spec_hit-base_hit:+d})   {sec()}")

assert not np.isnan(spec_oof[S_tr]).any(), "子人群内有样本未获得 OOF 预测"
assert np.isnan(spec_oof[~S_tr]).all(), "子人群外不应有预测"

print(f"\n【1】逐折增量: {[c - b for _, b, c in fold_stat]}  "
      f"合计 {sum(c - b for _, b, c in fold_stat):+d}")
ys, bs, ss = y[S_tr], s_tr[S_tr], spec_oof[S_tr]
print(f"  子人群内 AUC: 基线 {roc_auc_score(ys, bs):.4f} -> "
      f"专用 {roc_auc_score(ys, ss):.4f}")
bh = int(ys[np.argsort(-bs)[:m_quota]].sum())
sh = int(ys[np.argsort(-ss)[:m_quota]].sum())
print(f"  子人群内 top-{m_quota} 命中: 基线 {bh} -> 专用 {sh}（{sh-bh:+d}）"
      f"  上界 {min(m_quota, n_pos_S)}")

# ==================== 段三：保名额拼接与全局评估 ====================
print("\n" + "=" * 90)
print("【段三】保名额拼接后的全局评估")
print("=" * 90)


def splice(base, spec, S):
    """把 base 在 S 内的取值集合，按 spec 的顺序重新分配给 S 内的卡"""
    out = base.copy()
    idx = np.flatnonzero(S)
    order = idx[np.argsort(-spec[idx], kind="stable")]   # 专用模型从好到差
    vals = np.sort(base[idx])[::-1]                      # 基线取值从高到低
    out[order] = vals
    return out


new_tr = splice(s_tr, spec_oof, S_tr)
assert np.allclose(np.sort(new_tr[S_tr]), np.sort(s_tr[S_tr])), "S 内取值集合被改变"
assert np.array_equal(new_tr[~S_tr], s_tr[~S_tr]), "S 外分数被改动"
nrk = rankdata(-new_tr, method="ordinal")
nsel = nrk <= K_TR
assert int((nsel & S_tr).sum()) == m_quota, \
    f"名额未守恒: {int((nsel & S_tr).sum())} != {m_quota}"
print(f"  ✔ 名额守恒校验通过：子人群仍占 {m_quota} 个名额，子人群外分数未动")

tp_new = int(y[nsel].sum())
print(f"\n【1】全局 OOF top1%: {tp_base} -> {tp_new}（{tp_new-tp_base:+d}）")
b_idx, n_idx = set(np.flatnonzero(sel_tr)), set(np.flatnonzero(nsel))
sin, sout = sorted(n_idx - b_idx), sorted(b_idx - n_idx)
ip, op = int(y[sin].sum()), int(y[sout].sum())
print(f"【2/3】每侧换卡 m={len(sin)}  换入正例 {ip} / 换出正例 {op} "
      f"-> 净 {ip-op:+d}")
if sin:
    r = rk_tr[sin]
    print(f"  换入卡的基线原排名: min={r.min()} p50={int(np.median(r))} "
          f"max={r.max()}")
    rp = rk_tr[[i for i in sin if y[i] == 1]]
    if len(rp):
        print(f"  其中**新找回的正例**原排名: "
              f"min={rp.min()} p50={int(np.median(rp))} max={rp.max()}")
        for lo, hi in [(K_TR, 2 * K_TR), (2 * K_TR, 5 * K_TR),
                       (5 * K_TR, len(y) + 1)]:
            print(f"    [{lo},{hi}): {int(((rp>=lo)&(rp<hi)).sum())} 张")

print("\n【共同错误 / 互补错误】（子人群内，与基线对比）")
b_miss = set(np.flatnonzero((y == 1) & ~sel_tr & S_tr))
n_miss = set(np.flatnonzero((y == 1) & ~nsel & S_tr))
print(f"  基线漏报 {len(b_miss)} / 新漏报 {len(n_miss)}")
print(f"  共同漏报 {len(b_miss & n_miss)}  "
      f"仅基线漏(新找回) {len(b_miss - n_miss)}  "
      f"仅新漏(新丢失) {len(n_miss - b_miss)}")

print("\n【5】边界条件区分能力（基线排名落在该区间的子集内 AUC）")
for lo, hi in [(K_TR, 2 * K_TR), (int(0.5 * K_TR), 2 * K_TR)]:
    mask = (rk_tr >= lo) & (rk_tr < hi)
    if not (0 < y[mask].sum() < mask.sum()):
        continue
    print(f"  [{lo},{hi}) n={int(mask.sum())} 正例={int(y[mask].sum())}: "
          f"{roc_auc_score(y[mask], s_tr[mask]):.4f} -> "
          f"{roc_auc_score(y[mask], new_tr[mask]):.4f}")

print("\n【6】专用模型最看重什么（与全局模型的差异即为新学到的东西）")
imp_s = pd.Series(imp, index=cols).sort_values(ascending=False)
print("  专用模型 gain 前 15:")
for i, (c, v) in enumerate(imp_s.head(15).items()):
    print(f"    {i+1:>2}. {c:<30} {v/imp_s.sum():.2%}")
FAMS = {"快进快出/末段": ("fio", "fastout", "afterin", "trailing", "last5",
                          "burst", "h1_", "h24_"),
        "silence族": ("S_",),
        "财富归一": ("in_vs_aum", "bal_vs_aum", "inmax_vs_aum",
                     "in_retention", "sender_per_in"),
        "exp5暴露编码": EXP5_SUFFIX}
print("  各特征族在专用模型里的 gain 占比：")
for fam, pat in FAMS.items():
    if fam == "exp5暴露编码":
        cc = [c for c in cols if c.endswith(pat)]
    elif fam == "财富归一":
        cc = [c for c in cols if c in pat]
    else:
        cc = [c for c in cols if any(p in c for p in pat)]
    if cc:
        print(f"    {fam:<14} {len(cc):>3} 列  gain {imp_s[cc].sum()/imp_s.sum():.2%}")

# ==================== 段四：机构留出的独立方向性检验 ====================
print("\n" + "=" * 90)
print("【段四】机构留出验证（固定业务分组留出，非随机折）")
print("  ⚠ 该段**剔除全部 exp5 与 smy_te 列**，因为那些编码绑定 FOLD_SEED=42，")
print("    换折即泄露。剔除后模型更弱，但可独立检验『专用 > 全局』的方向性。")
print("=" * 90)
drop = [c for c in cols if c.endswith(EXP5_SUFFIX) or c.startswith("smy_te")]
free_cols = [c for c in cols if c not in set(drop)]
print(f"  剔除 {len(drop)} 列（exp5 {len([c for c in drop if not c.startswith('smy_te')])}"
      f" + smy_te {len([c for c in drop if c.startswith('smy_te')])}），"
      f"剩余 {len(free_cols)} 列")

card = pd.read_parquet(P("card.parquet"),
                       columns=["card_no", "crdisu_lvl1_insid"])
card["card_no"] = card["card_no"].astype(str)
g = tr[["card_no"]].merge(card, on="card_no", how="left")["crdisu_lvl1_insid"]
g = g.astype(str).fillna("__NA__").values
print(f"  发卡一级机构 {len(pd.unique(g))} 个，按机构做 {N_FOLDS} 组 GroupKFold")

gkf = list(GroupKFold(n_splits=N_FOLDS).split(Xtr, y, groups=g))
res = []
for k, (ti, vi) in enumerate(gkf):
    ti_s, vi_s = ti[S_tr[ti]], vi[S_tr[vi]]
    if len(vi_s) < 500 or y[vi_s].sum() < 10:
        print(f"  fold{k} 验证侧子人群样本/正例过少，跳过")
        continue
    mg = lgb.train(params, lgb.Dataset(Xtr.iloc[ti_s][free_cols], y[ti_s]),
                   3000, valid_sets=[lgb.Dataset(Xtr.iloc[vi_s][free_cols],
                                                 y[vi_s])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
    ps = mg.predict(Xtr.iloc[vi_s][free_cols], num_iteration=mg.best_iteration)
    # 同口径对照：同样剔列、同样 GroupKFold，但在**全量**上训练
    mf = lgb.train(params, lgb.Dataset(Xtr.iloc[ti][free_cols], y[ti]), 3000,
                   valid_sets=[lgb.Dataset(Xtr.iloc[vi][free_cols], y[vi])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
    pf = mf.predict(Xtr.iloc[vi_s][free_cols], num_iteration=mf.best_iteration)
    kq = max(int(len(vi_s) * TOP_FRAC), 1)
    hs = int(y[vi_s][np.argsort(-ps)[:kq]].sum())
    hf = int(y[vi_s][np.argsort(-pf)[:kq]].sum())
    res.append((hs, hf))
    print(f"  fold{k} 留出机构{len(pd.unique(g[vi]))}个 子人群验证{len(vi_s):>6,} "
          f"正例{int(y[vi_s].sum()):>4} 名额{kq:<4} "
          f"全量模型{hf:<4} 专用模型{hs:<4} ({hs-hf:+d})  "
          f"AUC {roc_auc_score(y[vi_s], pf):.4f}->{roc_auc_score(y[vi_s], ps):.4f}"
          f"   {sec()}")
if res:
    d = [a - b for a, b in res]
    print(f"\n  机构留出逐折增量 {d}  合计 {sum(d):+d}  "
          f"方向一致的折 {sum(1 for v in d if v > 0)}/{len(d)}")
    print("  判读：若方向与段二相反或正负各半，说明段二的增益依赖随机折结构，"
          "不可信")

# ==================== 段五：testB 侧 ====================
print("\n" + "=" * 90)
print(f"【4】{TARGET} 侧换卡")
print("=" * 90)
new_te = splice(s_te, spec_te, S_te)
assert np.allclose(np.sort(new_te[S_te]), np.sort(s_te[S_te]))
assert np.array_equal(new_te[~S_te], s_te[~S_te])
nsel_te = rankdata(-new_te, method="ordinal") <= K_TE
assert int((nsel_te & S_te).sum()) == int((sel_te & S_te).sum()), "testB 名额未守恒"
swap = int(K_TE - (nsel_te & sel_te).sum())
print(f"  子人群在 {TARGET} 头部占 {int((sel_te & S_te).sum())}/{K_TE} 个名额")
print(f"  top{K_TE} 换动 {swap} 张（换入=换出={swap}）")
tin = np.flatnonzero(nsel_te & ~sel_te)
if len(tin):
    print(f"  换入卡的基线原排名: min={rk_te[tin].min()} "
          f"p50={int(np.median(rk_te[tin]))} max={rk_te[tin].max()}")
    fv = pd.read_parquet(P("features_v4.parquet"),
                         columns=["card_no", "n_txn", "n_in", "aum_val",
                                  "n_fastout_1h", "out_ratio"])
    fv["card_no"] = fv["card_no"].astype(str)
    prof = te[["card_no"]].merge(fv, on="card_no", how="left")
    print("  换入卡 vs 基线头部（中位数），确认换的是不是目标人群：")
    for c in ["n_txn", "n_in", "aum_val", "n_fastout_1h", "out_ratio"]:
        a = np.nanmedian(prof[c].values[tin])
        b = np.nanmedian(prof[c].values[sel_te])
        print(f"    {c:<16} 换入 {a:>12,.2f}   基线头部 {b:>12,.2f}")

np.save(LP(f"oof_r01_{WHICH}_{TARGET}.npy"), new_tr)
np.save(LP(f"te_r01_{WHICH}_{TARGET}.npy"), new_te)
sub = te[["card_no"]].copy()
sub["score"] = new_te
assert (sub["card_no"].values == te["card_no"].values).all()
out = os.path.join(LAB_PRED, f"pred_{TARGET}_r01_{WHICH}.csv")
sub.to_csv(out, index=False)
with open(LP(f"r01_{WHICH}_{TARGET}.meta.json"), "w", encoding="utf-8") as f:
    json.dump({"划分": WHICH, "定义": DESC[WHICH], "fold_seed": FOLD_SEED,
               "子人群n": int(S_tr.sum()), "子人群pos": n_pos_S,
               "名额m": m_quota, "OOF基线": tp_base, "OOF新": tp_new,
               "换入正例": ip, "换出正例": op,
               f"{TARGET}换动": swap}, f, ensure_ascii=False, indent=2)
print(f"\n  候选文件（仅落盘，未提交）: {out}")
print(f"\n全部完成 {sec()}")
