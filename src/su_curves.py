# -*- coding: utf-8 -*-
"""§5.1 SU curves (Fig. f1): SU(A_i, d), A_i = first k attributes, k evenly
spaced (<=10 points per dataset — uniform rule replacing the old per-
dataset strides that broke on War). Descriptive statistic on the full data
(disclosed); writes curves/<abbr>.csv and curves_all.pdf."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 26, "axes.titlesize": 30,
                     "axes.labelsize": 26, "xtick.labelsize": 22,
                     "ytick.labelsize": 22})
from pathlib import Path
from load_dataset import load_dataset, column_codes
from harsugd_core import blocks_from_columns, symmetric_uncertainty

ORDER = ["Hil", "Sob", "Ger", "App", "Fla", "Spe", "Win", "Wpb", "Obs", "Dar",
         "Yal", "ORL40", "War", "Lun", "Pro", "Leu", "ORaw", "Mus"]

if __name__ == "__main__":
    Path("curves").mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",   # 数学体 $SU(P,d)$ 用 STIX，视觉对齐 Times
        "pdf.fonttype": 42,           # TrueType 嵌入 —— 期刊合规关键项
        "font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.linewidth": 0.8,
    })
    fig, axes = plt.subplots(6, 3, figsize=(10.5, 11.5))
    for i, (ax, abbr) in enumerate(zip(axes.ravel(), ORDER)):
        X, y, kinds = load_dataset(abbr)
        Xd, _ = column_codes(X, kinds)
        yb = pd.factorize(y)[0]
        m = Xd.shape[1]
        ks = np.unique(np.round(np.linspace(1, m, min(10, m))).astype(int))
        vals = [symmetric_uncertainty(blocks_from_columns(Xd[:, :k]), yb) for k in ks]
        ns = pd.read_csv("results_summary.csv")
        ns = ns[(ns.clf == "KNN") & (ns.method == "HAR-SUGD")].set_index("abbr")["n_sel"]
        pd.DataFrame({"k": ks, "SU": vals}).to_csv(f"curves/{abbr}.csv", index=False)
        ax.plot(ks, vals, "o-", markersize=6, linewidth=2)
        ax.axvline(ns[abbr], color="crimson", ls="--", lw=1.4, alpha=.85)
        ax.set_title(abbr)
        ax.set_xlabel("Number of attributes")     # label_outer 会隐藏非底行的
        ax.set_ylabel("$SU(P,d)$")                # label_outer 会隐藏非左列的
        ax.label_outer()                          # 只保留左列 y 标签、底行 x 标签
        ax.grid(alpha=.3)
    fig.tight_layout(h_pad=1.2, w_pad=1.8)
    fig.savefig("curves_all.pdf")
    print("curves/*.csv + curves_all.pdf written")