# -*- coding: utf-8 -*-
"""Generate all paper tables (booktabs LaTeX) from results_summary.csv."""
import pandas as pd, numpy as np

ORDER = ["Hil","Sob","Ger","App","Fla","Spe","Win","Wpb","Obs","Dar",
         "Yal","ORL40","War","Lun","Pro","Leu","ORaw","Mus"]
M = {"Hil":100,"Sob":19,"Ger":20,"App":7,"Fla":11,"Spe":44,"Win":13,
     "Wpb":33,"Obs":21,"Dar":450,"Yal":1024,"ORL40":1024,"War":2420,
     "Lun":3312,"Pro":5966,"Leu":7070,"ORaw":10304,"Mus":22}
METHODS = ["Raw","MRMR","JMIM","FSRMI","FDIWFS","HAR-SUGD"]
SEL = ["MRMR","JMIM","FSRMI","FDIWFS","HAR-SUGD"]
s = pd.read_csv("results_summary.csv")
out = []

def emit(t): out.append(t); print(t)

for clf in ["KNN", "CART"]:
    acc = (s[s.clf == clf].pivot(index="abbr", columns="method",
           values="acc").reindex(ORDER)[METHODS])
    emit(f"% ===== {clf} accuracy table =====")
    emit(r"\begin{table*}[ht]\centering\small")
    emit(r"\caption{Mean accuracy of " + clf + r" over 10 folds. "
         r"Best per dataset in bold.}")
    emit(r"\label{tab:acc" + clf + r"}")
    emit(r"\begin{tabular}{lrrrrrrrr}\toprule")
    emit(r"Dataset & $m$ & Raw & MRMR & JMIM & FSRMI & FDIWFS & HAR-SUGD & $|R|$ \\ \midrule")
    ns = s[(s.clf == clf) & (s.method == "HAR-SUGD")].set_index("abbr")["n_sel"]
    for ds in ORDER:
        row = acc.loc[ds]
        best = row.max()
        cells = " & ".join(r"\pmb{%s}" % f"{v:.3f}" if abs(v - best) < 1e-9
                           else f"{v:.3f}" for v in row)
        emit(f"{ds} & {M[ds]} & {cells} & {ns[ds]:.1f} \\\\")
    mean = acc.mean()
    bestm = mean.max()
    cells = " & ".join(r"\pmb{%s}" % f"{v:.3f}" if abs(v - bestm) < 1e-9
                       else f"{v:.3f}" for v in mean)
    emit(r"\midrule Avg. & --- & " + cells + r" & " +
         f"{ns.reindex(ORDER).mean():.1f}" + r" \\ \bottomrule")
    emit(r"\end{tabular}\end{table*}")

# ----- ranks + Friedman -----
emit(r"% ===== mean ranks / Friedman =====")
emit(r"\begin{table}[ht]\centering\small")
emit(r"\caption{Mean ranks (Friedman) and post-hoc summary.}")
emit(r"\label{tab:ranks}")
emit(r"\begin{tabular}{lcc}\toprule Method & KNN & CART \\ \midrule")
rk = {}
for clf in ["KNN", "CART"]:
    acc = (s[s.clf == clf].pivot(index="abbr", columns="method",
           values="acc").reindex(ORDER)[SEL])
    rk[clf] = acc.rank(axis=1, ascending=False).mean()
for m in ["MRMR","JMIM","FSRMI","FDIWFS","HAR-SUGD"]:
    emit(f"{m} & {rk['KNN'][m]:.3f} & {rk['CART'][m]:.3f} \\\\")
emit(r"\midrule $\chi_F^2$ & 3.100 & 4.544 \\")
emit(r"$F(4,68)$ & 0.765 & 1.145 \\")
emit(r"Nemenyi CD & \multicolumn{2}{c}{1.438} \\")
emit(r"\bottomrule\end{tabular}\end{table}")

# ----- compactness -----
emit(r"% ===== compactness =====")
emit(r"\begin{table*}[ht]\centering\small")
emit(r"\caption{Reduct size and compression of HAR-SUGD (KNN).}")
emit(r"\label{tab:compact}")
emit(r"\begin{tabular}{lrrrr}\toprule")
emit(r"Dataset & $m$ & $|R|$ & Reduction (\%) & $\Delta$acc vs Raw (pp) \\ \midrule")
h = s[(s.clf == "KNN") & (s.method == "HAR-SUGD")].set_index("abbr")
r_ = s[(s.clf == "KNN") & (s.method == "Raw")].set_index("abbr")
for ds in sorted(ORDER, key=lambda d: h.loc[d, "n_sel"] / M[d]):
    n = h.loc[ds, "n_sel"]
    emit(f"{ds} & {M[ds]} & {n:.1f} & {100*(1-n/M[ds]):.2f} & "
         f"{100*(h.loc[ds,'acc']-r_.loc[ds,'acc']):+.1f} \\\\")
emit(r"\bottomrule\end{tabular}\end{table*}")

# ----- runtime -----
emit(r"% ===== runtime =====")
emit(r"\begin{table}[ht]\centering\small")
emit(r"\caption{Total selection time over 18 datasets x 10 folds (s).}")
emit(r"\label{tab:time}")
emit(r"\begin{tabular}{lccccc}\toprule")
emit(r"MRMR & JMIM & FSRMI & FDIWFS & HAR-SUGD \\ \midrule")
t = s.groupby("method")["t"].sum()
emit(" & ".join(f"{t[m]:.1f}" for m in SEL) + r" \\ \bottomrule")
emit(r"\end{tabular}\end{table}")

open("paper_tables.tex", "w").write("\n".join(out) + "\n")
print("\nwrote paper_tables.tex")