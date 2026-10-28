# -*- coding: utf-8 -*-
"""analyze_final.py -- merged final analysis (FCBF + std columns).
Reads results_raw.csv (main experiment, untouched) + fcbf_raw.csv.
Prints: Tables 3/4 LaTeX bodies (mean±std, FCBF, bold best), Table 6
(|R|±std + FCBF |S|), ranks/Friedman/CD (k=6), Holm-corrected paired
tests vs HAR-SUGD (t + Wilcoxon), worst-case degradation, runtime,
eps=1e-3 breakdown. Prints only; writes nothing."""
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

ORDER = ["Hil","Sob","Ger","App","Fla","Spe","Win","Wpb","Obs",
         "Dar","Yal","ORL40","War","Lun","Pro","Leu","ORaw","Mus"]
METH  = ["Raw","MRMR","JMIM","FCBF","FSRMI","FDIWFS","HAR-SUGD"]
SEL   = ["MRMR","JMIM","FCBF","FSRMI","FDIWFS","HAR-SUGD"]
M_ATTR = {"Hil":100,"Sob":19,"Ger":20,"App":7,"Fla":11,"Spe":44,"Win":13,
          "Wpb":33,"Obs":21,"Dar":450,"Yal":1024,"ORL40":1024,"War":2420,
          "Lun":3312,"Pro":5966,"Leu":7070,"ORaw":10304,"Mus":22}

a = pd.read_csv("results_raw.csv")[["abbr","fold","clf","method","acc","n_sel","t"]]
b = pd.read_csv("fcbf_raw.csv")[["abbr","fold","clf","method","acc","n_sel","t"]]
df = pd.concat([a, b], ignore_index=True)
summ = df.groupby(["clf","abbr","method"]).agg(
    acc=("acc","mean"), std=("acc","std"),
    n=("n_sel","mean"), nstd=("n_sel","std")).reset_index()

def holm(pvals):
    m = len(pvals); order = np.argsort(pvals); adj = np.empty(m); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * pvals[idx]); adj[idx] = min(1.0, run)
    return adj

for clf in ["KNN", "CART"]:
    sub = summ[summ.clf == clf]
    piv = sub.pivot(index="abbr", columns="method", values="acc").reindex(ORDER)[METH]

    # ---- Table 3/4 LaTeX body: mean±std, bold best (rounded to 3dp) ----
    print(f"\n===== {clf}: Tables 3/4 body (mean$\\pm$std, best bold) =====")
    for ds in ORDER:
        row = piv.loc[ds]; mx = round(row.max(), 3)
        cells = []
        for mth in METH:
            s = sub[(sub.abbr == ds) & (sub.method == mth)]["std"].iloc[0]
            v = round(row[mth], 3)
            c = f"{v:.3f}$\\pm${s:.3f}"
            cells.append(f"\\pmb{{{c}}}" if v >= mx else c)
        nr = sub[(sub.abbr == ds) & (sub.method == "HAR-SUGD")].n.iloc[0]
        print(f"{ds} & {M_ATTR[ds]} & " + " & ".join(cells) + f" & {nr:.1f} \\\\")
    print("Avg. & --- & " + " & ".join(f"{piv[m].mean():.3f}" for m in METH) + " & "
          f"{summ[(summ.clf==clf)&(summ.method=='HAR-SUGD')].n.mean():.1f} \\\\")

    # ---- Friedman k=6 ----
    acc6 = piv[SEL]
    ranks = acc6.rank(axis=1, ascending=False).mean(axis=0)
    k, N = 6, 18
    chi2 = 12.0*N/(k*(k+1))*((ranks**2).sum() - k*(k+1)**2/4)
    F = (N-1)*chi2/(N*(k-1)-chi2)
    CD = 2.850*np.sqrt(k*(k+1)/(6.0*N))
    print(f"\n--- {clf}: mean ranks ---")
    print(ranks.round(3).to_string())
    print(f"chi2={chi2:.3f}  F(5,85)={F:.3f}  CD={CD:.3f}  "
          f"max rank gap={ranks.max()-ranks.min():.3f}")

    # ---- paired tests vs HAR-SUGD, Holm over 10 ----
    labels, pvals = [], []
    print(f"\n--- {clf}: paired tests vs HAR-SUGD (raw p) ---")
    for mth in ["MRMR","JMIM","FCBF","FSRMI","FDIWFS"]:
        _, pt = ttest_rel(piv["HAR-SUGD"], piv[mth])
        try:
            _, pw = wilcoxon(piv["HAR-SUGD"], piv[mth])
        except ValueError:
            pw = 1.0
        labels += [f"t:{mth}", f"W:{mth}"]; pvals += [pt, pw]
        print(f"  {mth:7s} t-p={pt:.4f}  Wilcoxon-p={pw:.4f}")
    adj = holm(np.array(pvals))
    print("Holm-adjusted (min):", adj.min().round(4),
          "| uncorrected<0.05:",
          [l for l, p in zip(labels, pvals) if p < 0.05])
    print("survive correction:", sum(adj < 0.05))

    # ---- worst-case degradation vs Raw ----
    wc = {mth: (piv[mth] - piv["Raw"]).min() for mth in SEL}
    print(f"--- {clf}: worst-case degradation (pp) ---")
    print({m: round(v*100, 1) for m, v in sorted(wc.items(), key=lambda x: -x[1])})

# ---- Table 6: |R|±std (HAR) + FCBF |S| ----
print("\n===== Table 6: |R|±std + FCBF |S| =====")
h = summ[(summ.clf == "KNN") & (summ.method == "HAR-SUGD")].set_index("abbr")
f = summ[(summ.clf == "KNN") & (summ.method == "FCBF")].set_index("abbr")
for ds in ORDER:
    print(f"{ds} & {M_ATTR[ds]} & {h.n[ds]:.1f}±{h.nstd[ds]:.1f} & {f.n[ds]:.1f} \\\\")

# ---- runtime totals (dedupe: t logged once per fold per method) ----
tt = df.drop_duplicates(["abbr", "fold", "method"]).groupby("method").t.sum()
print("\n===== Table 7: total selection time (s) =====")
print({m: round(tt[m], 1) for m in ["MRMR","JMIM","FCBF","FSRMI","FDIWFS","HAR-SUGD"]})

# ---- eps=1e-3 breakdown ----
try:
    e = pd.DataFrame(json.load(open("eps_check.json")))
    print("\n eps=1e-3 differing folds by dataset:")
    print(e[~e.same_1e_3].abbr.value_counts().to_string())
except FileNotFoundError:
    pass