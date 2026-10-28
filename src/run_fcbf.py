# -*- coding: utf-8 -*-
"""FCBF surgical driver: adds ONLY the FCBF baseline. Reuses the main
experiment's pipeline verbatim (folds/*.csv, fold-internal imputation,
column_codes, KNN/CART, one-hot). Writes fcbf_raw.csv (+ delta sweep ->
fcbf_delta.json). Never touches results_raw.csv / main driver. Row format
matches results_raw.csv so both files concatenate for analysis."""
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

from load_dataset import load_dataset, column_codes
from harsugd_core import symmetric_uncertainty, refine

SEED = 42
ORDER = ["Hil", "Sob", "Ger", "App", "Fla", "Spe", "Win", "Wpb", "Obs",
         "Dar", "Yal", "ORL40", "War", "Lun", "Pro", "Leu", "ORaw", "Mus"]
DELTA = 0.5          # FCBF threshold (per reviewer's request)
SWEEP = [0.3, 0.7]   # sensitivity sweep, selection only


# ---------- FCBF (Yu & Liu 2004), on integer codes ----------
def fcbf_select(Xd, ytr, delta=0.5):
    """Returns (selected, prefilter_empty). Empty pre-filter lets the
    caller apply the main driver's last-resort guard."""
    yb = pd.factorize(ytr)[0]
    n, m = Xd.shape
    zeros = np.zeros(n, dtype=np.int64)
    parts = [refine(zeros, Xd[:, j]) for j in range(m)]        # U/{a_j}
    su_d = np.array([symmetric_uncertainty(parts[j], yb) for j in range(m)])
    S = [j for j in range(m) if su_d[j] > delta]               # pre-filter
    if not S:
        return [], True
    S.sort(key=lambda j: (-su_d[j], j))                        # SU desc
    out = []
    while S:                                                   # dominance
        f = S.pop(0)
        out.append(f)
        S = [g for g in S
             if symmetric_uncertainty(parts[f], parts[g]) < su_d[g]]
    return out, False


# ---------- copied verbatim from the main driver ----------
def make_cv(y, n_splits=10):
    counts = np.bincount(pd.factorize(y)[0])
    if counts.min() >= n_splits:
        return StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    return KFold(n_splits, shuffle=True, random_state=SEED)

def clf_matrices(Xtr, Xte, kinds, sel):
    num = [j for j in sel if kinds[j] == "num"]
    sym = [j for j in sel if kinds[j] == "sym"]
    Atr = Xtr.iloc[:, num].astype(float).values if num else np.zeros((len(Xtr), 0))
    Ate = Xte.iloc[:, num].astype(float).values if num else np.zeros((len(Xte), 0))
    if sym:
        oh = OneHotEncoder(handle_unknown="ignore")
        Htr = oh.fit_transform(Xtr.iloc[:, sym].astype(str))
        Hte = oh.transform(Xte.iloc[:, sym].astype(str))
        Htr = Htr.toarray() if hasattr(Htr, "toarray") else np.asarray(Htr)
        Hte = Hte.toarray() if hasattr(Hte, "toarray") else np.asarray(Hte)
        Atr = np.hstack([Atr, Htr]); Ate = np.hstack([Ate, Hte])
    return Atr, Ate

def get_splits(X, y, abbr):
    f = Path(f"folds/{abbr}.csv")
    if f.exists():
        fold_id = pd.read_csv(f, index_col=0).iloc[:, 0].values
        return [(np.where(fold_id != k)[0], np.where(fold_id == k)[0])
                for k in range(10)]
    return list(make_cv(y).split(X, y))


# ---------- the run ----------
def run_dataset(abbr):
    X, y, kinds = load_dataset(abbr)
    m = X.shape[1]
    recs, sweep = [], []
    for fold, (tr, te) in enumerate(get_splits(X, y, abbr)):
        Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
        ytr, yte = y.iloc[tr].values, y.iloc[te].values
        for c, k_ in zip(X.columns, kinds):        # same imputation as main
            if Xtr[c].isna().any() or Xte[c].isna().any():
                fill = Xtr[c].median() if k_ == "num" else Xtr[c].mode()[0]
                Xtr[c] = Xtr[c].fillna(fill); Xte[c] = Xte[c].fillna(fill)
        Xd, _ = column_codes(Xtr, kinds)

        t0 = time.perf_counter()
        sel, empty = fcbf_select(Xd, ytr, delta=DELTA)
        t_sel = time.perf_counter() - t0
        guard = False
        if empty:                                  # main driver's last-resort
            yb = pd.factorize(ytr)[0]
            su1 = [symmetric_uncertainty(refine(np.zeros(len(yb), dtype=np.int64),
                   Xd[:, j]), yb) for j in range(m)]
            sel = [int(np.argmax(su1))]
            guard = True
        sel_str = ",".join(map(str, sel))

        for d in SWEEP:                            # sweep: selection only
            s2, e2 = fcbf_select(Xd, ytr, delta=d)
            sweep.append(dict(abbr=abbr, fold=fold, delta=d, n_sel=len(s2),
                              sel=",".join(map(str, s2)), empty=e2))

        for clf_name, mk in [("KNN", lambda: KNeighborsClassifier(n_neighbors=3)),
                             ("CART", lambda: DecisionTreeClassifier(random_state=SEED))]:
            Atr, Ate = clf_matrices(Xtr, Xte, kinds, sel)
            recs.append(dict(abbr=abbr, fold=fold, clf=clf_name, method="FCBF",
                             acc=mk().fit(Atr, ytr).score(Ate, yte),
                             n_sel=len(sel), t=t_sel, sel=sel_str, guard=guard))
        print(f"  {abbr} fold {fold+1}/10 done", flush=True)
    return recs, sweep


if __name__ == "__main__":
    allrec, sweep = [], []
    for abbr in ORDER:
        print("FCBF:", abbr, flush=True)
        try:
            r, s = run_dataset(abbr)
            allrec += r; sweep += s
        except Exception:
            import traceback; traceback.print_exc()
            print(f"!! {abbr} FAILED -- continuing", flush=True)
        pd.DataFrame(allrec).to_csv("fcbf_raw.csv", index=False)   # checkpoint
        json.dump(sweep, open("fcbf_delta.json", "w"))
    print(f"\nDone: {len(allrec)} rows -> fcbf_raw.csv, "
          f"{len(sweep)} sweep records -> fcbf_delta.json")