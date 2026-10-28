# stats_upgrade.py
import pandas as pd, numpy as np
from scipy.stats import ttest_rel, wilcoxon
ORDER = ["Hil","Sob","Ger","App","Fla","Spe","Win","Wpb","Obs","Dar",
         "Yal","ORL40","War","Lun","Pro","Leu","ORaw","Mus"]
s = pd.read_csv("results_summary.csv")
for clf in ["KNN", "CART"]:
    acc = s[s.clf == clf].pivot(index="abbr", columns="method", values="acc").reindex(ORDER)
    print(f"\n===== {clf} (Holm over all tests) =====")
    ps = []
    for m in ["MRMR","JMIM","FSRMI","FDIWFS"]:
        _, p = ttest_rel(acc["HAR-SUGD"], acc[m]); ps.append(("t", m, p))
        try:
            _, pw = wilcoxon(acc["HAR-SUGD"], acc[m]); ps.append(("W", m, pw))
        except ValueError: pass
    ps.sort(key=lambda x: x[2]); n = len(ps)
    adj = np.minimum(np.maximum.accumulate([p*(n-i) for i,(k,m,p) in enumerate(ps)]), 1.0)
    for (k, m, p), a in zip(ps, adj):
        print(f"  {k} vs {m:6s}: p={p:.4f}  Holm-adj={a:.4f}" + ("  *" if a < 0.05 else ""))