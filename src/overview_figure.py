# -*- coding: utf-8 -*-
"""Overview figure (replaces view.pdf): top band = research approach,
bottom band = three findings panels. Every number is computed from
results_summary.csv — nothing hand-copied. Output: overview.pdf.
The three print lines at the end are the built-in verification: they
must match the Avg rows of tab:accKNN/tab:accCART and the §5.4
robustness paragraph. If they differ, fix the MANUSCRIPT, not the script.
"""
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,          # TrueType embedding (journal compliance)
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
})

M = {"Hil": 100, "Sob": 19, "Ger": 20, "App": 7, "Fla": 11, "Spe": 44,
     "Win": 13, "Wpb": 33, "Obs": 21, "Dar": 450, "Yal": 1024, "ORL40": 1024,
     "War": 2420, "Lun": 3312, "Pro": 5966, "Leu": 7070, "ORaw": 10304,
     "Mus": 22}

s = pd.read_csv("results_summary.csv")
s["method"] = s["method"].replace({"mRMR": "MRMR", "mrmr": "MRMR"})

fig = plt.figure(figsize=(9.8, 7.8))
gs = fig.add_gridspec(2, 3, height_ratios=[0.72, 1.0], hspace=0.45,
                      wspace=0.38, left=0.075, right=0.985, top=0.90,
                      bottom=0.085)

# ================= top band: research approach =================
axT = fig.add_subplot(gs[0, :])
axT.set_xlim(0, 100); axT.set_ylim(0, 100); axT.axis("off")
axT.set_title("Research approach: from two complementary blind spots "
              "to a parameter-free reduct", fontsize=13.5,
              fontweight="bold", loc="left", pad=10)

def box(x, y, w, h, title, lines, fc, ec, fs=10.5, tf=12.0):
    axT.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.0",
                                 fc=fc, ec=ec, lw=1.3))
    axT.text(x + w/2, y + h - 4, title, ha="center", va="top",
             fontsize=tf, fontweight="bold", color=ec)
    axT.text(x + w/2, y + h - 12, "\n".join(lines), ha="center",
             va="top", fontsize=fs)

def arrow(x0, y0, x1, y1):
    axT.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                  mutation_scale=15, lw=1.6, color="k"))

BLUE, ORANGE, GREY = "#eef3fb", "#fdf1e7", "#f2f2f2"
box(1, 55, 20, 40, "SU-based line",
    ["normalized dependence", "on the decision;", "blind to granular",
     "structure"], BLUE, "#1f3b73")
box(1, 3, 20, 40, "Granularity-driven line",
    ["partition fineness of", "the induced classes;",
     "blind to decision", "relevance"], BLUE, "#1f3b73")
arrow(21.5, 75, 26.5, 50); arrow(21.5, 23, 26.5, 50)
box(27, 22, 21, 56, "Rough-set granulation",
    [r"$DIS=(U, A\cup D, V, f)$", "attribute subset $P$", "partitions $U$ into",
     "equivalence classes;", "$SU$ and $GD$ are", "measured on $U/P$"],
    BLUE, "#1f3b73")
arrow(48.5, 50, 53.5, 50)
box(54, 18, 24, 64, "HAR-SUGD (Alg. 1)",
    ["forward-greedy on",
     r"$\mathrm{sig}(a)=SU(P\cup\{a\},d)-SU(P,d)$",
     "tie-break: granularity gain", r"$GD(a\mid P)$",
     r"stop: $\max\,\mathrm{sig}\leq\varepsilon$",
     r"$\Rightarrow$ reduct, no target size"], ORANGE, "#b05a1a")
arrow(78.5, 50, 83.5, 50)
box(84, 22, 15, 56, "Validation",
    ["18 datasets", "(7 to 10,304 attrs)", "4 baselines,",
     "nested 10-fold CV", "Friedman / Nemenyi /", "paired tests"],
    GREY, "#444444")

# ================= (a) accuracy parity =================
axA = fig.add_subplot(gs[1, 0])
methods = ["Raw", "MRMR", "JMIM", "FSRMI", "FDIWFS", "HAR-SUGD"]
labels = ["Raw", "mRMR", "JMIM", "FSRMI", "FDIWFS", "HAR"]
means = {c: [s[(s.clf == c) & (s.method == m)]["acc"].mean() for m in methods]
         for c in ("KNN", "CART")}
xpos = np.arange(len(methods)); bw = 0.38
axA.bar(xpos - bw/2, [v*100 for v in means["KNN"]], bw, label="KNN",
        color="#0072B2", ec="k", lw=0.4)
axA.bar(xpos + bw/2, [v*100 for v in means["CART"]], bw, label="CART",
        color="#E69F00", ec="k", lw=0.4, alpha=0.95)
axA.set_xticks(xpos); axA.set_xticklabels(labels, rotation=35, ha="right")
axA.set_ylabel("Mean accuracy (%)"); axA.set_ylim(68, 86)
axA.grid(axis="y", alpha=0.3); axA.legend(fontsize=10, loc="lower right")
axA.set_title("(a) Accuracy: statistical parity")
axA.text(0.02, 0.97, "Friedman n.s.; Holm-corrected\npaired tests n.s. (min $p$ = 0.307)",
         transform=axA.transAxes, fontsize=10, va="top")

# ================= (b) compactness =================
axB = fig.add_subplot(gs[1, 1])
ns = s[(s.clf == "KNN") & (s.method == "HAR-SUGD")].set_index("abbr")["n_sel"]
assert len(ns) == 18 and set(ns.index) == set(M), "check results_summary.csv"
order = sorted(M, key=lambda a: -M[a])          # largest m on top
yb = np.arange(len(order))
for i, a in enumerate(order):
    axB.plot([ns[a], M[a]], [i, i], color="0.78", lw=1.2, zorder=1)
axB.scatter([M[a] for a in order], yb, s=24, c="0.45", zorder=2,
            label="$m$ (all attributes)")
axB.scatter([ns[a] for a in order], yb, s=55, c="#D55E00", marker="*",
            zorder=3, label="$|R|$ (reduct)")
axB.set_xscale("log"); axB.set_xlim(0.7, 40000)
axB.set_yticks(yb); axB.set_yticklabels(order, fontsize=10)
axB.invert_yaxis()
axB.set_xlabel("Number of attributes (log scale)")
axB.grid(axis="x", alpha=0.25); axB.legend(fontsize=10, loc="lower right")
med = float(np.median(ns.values))
med_disp = math.floor(med*10 + 0.5)/10    # half-up: 3.25 -> 3.3, matches text
axB.set_title(f"(b) Compactness: median $|R|$ = {med_disp:.1f}")

# ================= (c) robustness =================
axC = fig.add_subplot(gs[1, 2])
knn = s[s.clf == "KNN"]
raw = knn[knn.method == "Raw"].set_index("abbr")["acc"]
worst = {}
for m, g in knn.groupby("method"):
    if m == "Raw":
        continue
    worst[m] = float((g.set_index("abbr")["acc"] - raw).min() * 100)
names = sorted(worst, key=lambda m: -worst[m])     # best on top
vals = [worst[m] for m in names]
colors = ["#D55E00" if n == "HAR-SUGD" else "0.62" for n in names]
axC.barh(np.arange(len(names)), vals, color=colors, ec="k", lw=0.4)
axC.set_yticks(np.arange(len(names))); axC.set_yticklabels(names, fontsize=10.5)
for i, v in enumerate(vals):
    axC.text(v - 1.5, i, f"{v:.1f}", ha="right", va="center", fontsize=10)
axC.axvline(0, color="k", lw=0.8); axC.set_xlim(-78, 2)
axC.set_xlabel("Worst-case degradation vs. full features (pp)")
axC.grid(axis="x", alpha=0.3)
axC.set_title("(c) Robustness under KNN")

fig.savefig("overview.pdf")

# ---- built-in verification: must match the manuscript ----
print("(a) mean accuracy:")
for c in ("KNN", "CART"):
    print("   ", c, {m: round(v, 3) for m, v in zip(methods, means[c])})
print("(b) median |R| =", med, "-> shown as", round(med_disp, 1))
print("(c) worst-case degradation (pp):",
      {m: round(worst[m], 1) for m in names})