# -*- coding: utf-8 -*-
"""07 暴露编码 exp5（56 维，8 变体 × 7 统计）

核心：把「该卡的交易对手，在参考集里关联过多少张已知欺诈卡」编码成特征。
必须 fold-safe —— 每折的编码只使用其它 4 折的 label。

⚠ 本步含一处相对赛时初版的**口径修正**（见说明文档 §一.3）：
   初版 train 侧编码 ref = 4/5 折（240k 卡 / 2400 正），
   而 test 侧 ref = 全量 train（300k 卡 / 3000 正），标签密度差 1.25 倍，
   使 nf_max 一类计数型统计在 test 侧被系统性放大 24.6%。
   本版改为 **fold-matched**：折 k 的模型使用「以折 k 的 4/5 训练折为 ref」
   重算的 test 编码，两侧标签密度完全一致。
   验证：仅用 56 维 exp5 区分 train / testA 的对抗 AUC 由 0.9695 降至 0.4989。

输出:
  data/interim/exp5_tr.parquet          训练侧 fold-safe OOF 编码
  data/interim/exp5_te_f{0..4}.parquet  测试侧 fold-matched 编码（5 份）
耗时: 约 8 分钟
"""
import os
import sys
import time

import polars as pl
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P, FOLD_SEED, N_FOLDS

t0 = time.time()
TARGET = sys.argv[1] if len(sys.argv) > 1 else "testa"   # testa / testb


# ---------------- 8 个边表变体 ----------------
def build_variants():
    ea = pl.read_parquet(P("edge_acc2.parquet"))
    en = pl.read_parquet(P("edge_nm2.parquet"))
    # 低度数键：对手连接的卡数 <= 100（枢纽账号在其上，度数 q0.999≈66）
    la = (ea.group_by("key").agg(pl.col("card_no").n_unique().alias("deg"))
          .filter(pl.col("deg") <= 100).select("key"))
    ln = (en.group_by("key").agg(pl.col("card_no").n_unique().alias("deg"))
          .filter(pl.col("deg") <= 100).select("key"))

    def prep(e, a):
        return e.with_columns(pl.col(a).alias("amt"))

    return [
        ("aa",  prep(ea, "out_sum")),                                  # 账号-全部
        ("ao",  prep(ea.filter(pl.col("out_n") > 0), "out_sum")),      # 账号-出金
        ("ai",  prep(ea.filter(pl.col("in_n") > 0), "in_sum")),        # 账号-入金
        ("aol", prep(ea.filter(pl.col("out_n") > 0), "out_sum")
                .join(la, on="key", how="semi")),                      # 出金-低度数
        ("aop", prep(ea.filter((pl.col("out_n") > 0)
                               & (pl.col("prvt") == 1)), "out_sum")),  # 出金-对私
        ("na",  prep(en, "out_sum")),                                  # 名称-全部
        ("no",  prep(en.filter(pl.col("out_n") > 0), "out_sum")),      # 名称-出金
        ("nol", prep(en.filter(pl.col("out_n") > 0), "out_sum")
                .join(ln, on="key", how="semi")),                      # 名称出金-低度数
    ]


def expo(edges, tgt, ref, prefix):
    """tgt: 待编码的卡（只含 card_no）; ref: 参考集（card_no + label）"""
    cp = (edges.join(ref, on="card_no", how="inner")
          .group_by("key").agg([
              pl.col("card_no").filter(pl.col("label") == 1)
                .n_unique().alias("nf"),          # 该对手关联的欺诈卡数
              pl.col("card_no").n_unique().alias("nc")]))   # 关联的参考卡总数
    e = (edges.join(tgt, on="card_no", how="inner")
         .join(cp, on="key", how="left")
         .with_columns([pl.col("nf").fill_null(0), pl.col("nc").fill_null(0)]))
    e = e.with_columns((pl.col("nf") / (pl.col("nc") + 1)).alias("rate"))
    return e.group_by("card_no").agg([
        pl.col("nf").max().alias(f"{prefix}_nf_max"),
        (pl.col("nf") >= 1).sum().alias(f"{prefix}_ncp_f1"),
        (pl.col("nf") >= 2).sum().alias(f"{prefix}_ncp_f2"),
        pl.col("rate").max().alias(f"{prefix}_rate_max"),
        ((pl.col("rate") * pl.col("n")).sum() / pl.col("n").sum())
            .alias(f"{prefix}_rate_wavg"),
        pl.col("amt").filter(pl.col("nf") >= 1).sum().alias(f"{prefix}_amt_f"),
        (pl.col("n").filter(pl.col("nf") >= 1).sum() / pl.col("n").sum())
            .alias(f"{prefix}_txnf_share")])


def expo_all(V, tgt, ref):
    out = None
    for pfx, e in V:
        g = expo(e, tgt, ref, pfx)
        out = g if out is None else out.join(g, on="card_no", how="full",
                                             coalesce=True)
    return out


def main():
    train = pl.read_parquet(P("train.parquet"))
    tgt_tbl = pl.read_parquet(P(f"{TARGET}.parquet")).select("card_no")
    train_pl = train.select(["card_no", "label"]).with_columns(
        pl.col("label").cast(pl.Int32))
    y = train["label"].to_numpy()

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True,
                                 random_state=FOLD_SEED).split(train, y))
    V = build_variants()
    print(f"VARIANTS 就绪 {time.time()-t0:.0f}s")

    # ---- 训练侧：fold-safe OOF 编码 ----
    p_tr = P("exp5_tr.parquet")
    if not os.path.exists(p_tr):
        parts = [expo_all(V, train_pl[i_va].select("card_no"), train_pl[i_tr])
                 for i_tr, i_va in folds]
        pl.concat(parts).write_parquet(p_tr)
        print(f"exp5_tr 完成 {time.time()-t0:.0f}s")
    else:
        print("exp5_tr 已存在，跳过")

    # ---- 测试侧：fold-matched（5 份，ref 与各折训练集一致）----
    pfx = "exp5_te" if TARGET == "testa" else "exp5_teb"
    for k, (i_tr, _) in enumerate(folds):
        p = P(f"{pfx}_f{k}.parquet")
        if os.path.exists(p):
            continue
        expo_all(V, tgt_tbl, train_pl[i_tr]).write_parquet(p)
        print(f"  fold{k}  ref={len(i_tr):,} 卡 / {int(y[i_tr].sum())} 正   "
              f"{time.time()-t0:.0f}s")
    print(f"07 完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
