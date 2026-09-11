# -*- coding: utf-8 -*-
"""07b smy_cd 加权目标编码（1 维，fold-safe + fold-matched）

验证依据（见 code/explore/{field_value_check,smy_te_corr_check,smy_te_lift_check}.py）：
  · 单变量 AUC = 0.7714，与已有 299 维特征最高相关仅 0.4767（低于 silence
    族当初 0.491 的参考线）→ 判定为现有 Top95 smy one-hot 展开未覆盖的新信息
  · 单一 LightGBM 5 折 OOF 快速验证：401 维 -> 402 维，top1_f1 0.9163 -> 0.9190，
    命中 2749 -> 2757（+8）

编码定义：卡内每笔交易的 smy_cd，按参考集算出的历史欺诈率，
按该卡各码值出现次数加权平均成一个数值。

⚠ 与 exp5 完全相同的 fold-safe / fold-matched 要求（同一份折种子 FOLD_SEED）：
  训练侧用 5 折 OOF；测试侧对每一折的模型，用该折自己的 4/5 训练折重算
  test 编码（避免 exp5 初版犯过的参考集规模不一致、标签密度错位问题）。

输出:
  data/interim/smy_te_tr.parquet          训练侧 fold-safe OOF 编码
  data/interim/smy_te_te_f{0..4}.parquet  测试侧 fold-matched 编码（5 份）
耗时: 约 1~2 分钟
"""
import os
import sys
import time

import pandas as pd
import polars as pl
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P, FOLD_SEED, N_FOLDS

t0 = time.time()
TARGET = sys.argv[1] if len(sys.argv) > 1 else "testa"   # testa / testb


def main():
    train = pl.read_parquet(P("train.parquet")).select(["card_no", "label"]).to_pandas()
    train["card_no"] = train["card_no"].astype(str)
    train["label"] = train["label"].astype(int)
    y = train["label"].values
    global_mean = y.mean()
    label_map = dict(zip(train["card_no"], train["label"]))

    txn = pl.read_parquet(P("txn.parquet"), columns=["card_no", "smy_cd"])
    smy_all = (txn.group_by(["card_no", "smy_cd"]).len()
               .rename({"len": "n"}).to_pandas())
    smy_all["card_no"] = smy_all["card_no"].astype(str)
    del txn

    tgt_tbl = pl.read_parquet(P(f"{TARGET}.parquet")).select("card_no").to_pandas()
    tgt_tbl["card_no"] = tgt_tbl["card_no"].astype(str)

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True,
                                 random_state=FOLD_SEED).split(train, y))

    def code_rate_from(ref_cards):
        ref = smy_all[smy_all["card_no"].isin(ref_cards)].copy()
        ref["label"] = ref["card_no"].map(label_map)
        ref["wl"] = ref["label"] * ref["n"]
        stat = ref.groupby("smy_cd").agg(w=("wl", "sum"), n=("n", "sum"))
        return (stat["w"] / stat["n"]).to_dict()

    def encode(cards, code_rate):
        e = smy_all[smy_all["card_no"].isin(cards)].copy()
        e["rate"] = e["smy_cd"].map(code_rate).fillna(global_mean)
        e["wn"] = e["rate"] * e["n"]
        agg = e.groupby("card_no").agg(wn_sum=("wn", "sum"), n_sum=("n", "sum"))
        agg["smy_te"] = agg["wn_sum"] / agg["n_sum"]
        out = agg[["smy_te"]].reset_index()
        missing = pd.DataFrame({"card_no": list(set(cards) - set(out["card_no"]))})
        if len(missing):
            missing["smy_te"] = global_mean
            out = pd.concat([out, missing], ignore_index=True)
        return out

    # ---- 训练侧：fold-safe OOF ----
    p_tr = P("smy_te_tr.parquet")
    if not os.path.exists(p_tr):
        parts = []
        for i_tr, i_va in folds:
            tr_cards = set(train.iloc[i_tr]["card_no"])
            va_cards = set(train.iloc[i_va]["card_no"])
            code_rate = code_rate_from(tr_cards)
            parts.append(encode(va_cards, code_rate))
        pd.concat(parts, ignore_index=True).to_parquet(p_tr, index=False)
        print(f"smy_te_tr 完成 {time.time()-t0:.0f}s")
    else:
        print("smy_te_tr 已存在，跳过")

    # ---- 测试侧：fold-matched（5 份，ref 与各折训练集一致）----
    pfx = "smy_te_te" if TARGET == "testa" else "smy_te_teb"
    tgt_cards = set(tgt_tbl["card_no"])
    for k, (i_tr, _) in enumerate(folds):
        p = P(f"{pfx}_f{k}.parquet")
        if os.path.exists(p):
            continue
        tr_cards = set(train.iloc[i_tr]["card_no"])
        code_rate = code_rate_from(tr_cards)
        encode(tgt_cards, code_rate).to_parquet(p, index=False)
        print(f"  fold{k} 完成 {time.time()-t0:.0f}s")

    print(f"07b 完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
