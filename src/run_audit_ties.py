# -*- coding: utf-8 -*-
"""W-R1 tie audit: replays the main experiment's HAR-SUGD selection on the
SAME folds (folds/*.csv), SAME imputation, SAME column_codes pipeline.
Baselines and classifiers stripped -- only selection trajectories.
Writes tie_audit.json (checkpoint per dataset). Never touches
results_raw.csv / results_summary.csv."""
import json, time
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold

from load_dataset import load_dataset, column_codes
from harsugd_core import har_sugd, symmetric_uncertainty, refine

SEED = 42
ORDER = ["Hil", "Sob", "Ger", "App", "Fla", "Spe", "Win", "Wpb", "Obs",
         "Dar", "Yal", "ORL40", "War", "Lun", "Pro", "Leu", "ORaw", "Mus"]

def make_cv(y, n_splits=10):                       # copied verbatim from main
    counts = np.bincount(pd.factorize(y)[0])
    if counts.min() >= n_splits:
        return StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    return KFold(n_splits, shuffle=True, random_state=SEED)

def audit_dataset(abbr):
    X, y, kinds = load_dataset(abbr)
    m = X.shape[1]
    audit = []
    # --- same fold logic as main driver: reuse persisted folds ---
    f = Path(f"folds/{abbr}.csv")
    if f.exists():
        fold_id = pd.read_csv(f, index_col=0).iloc[:, 0].values
        splits = [(np.where(fold_id != k)[0], np.where(fold_id == k)[0])
                  for k in range(10)]
    else:
        splits = list(make_cv(y).split(X, y))
    for fold, (tr, te) in enumerate(splits):
        Xtr = X.iloc[tr].copy()
        ytr = y.iloc[tr].values
        for c, k_ in zip(X.columns, kinds):        # same fold-internal impute
            if Xtr[c].isna().any():
                fill = Xtr[c].median() if k_ == "num" else Xtr[c].mode()[0]
                Xtr[c] = Xtr[c].fillna(fill)
        Xd, _ = column_codes(Xtr, kinds)           # same discretization
        rec = []
        sel_h = har_sugd(Xd, ytr, log=rec)             # logged trajectory
        sel_nog = har_sugd(Xd, ytr, tie_break="none")  # no-GD ablation
        if not sel_h:                               # same guard as main
            yb = pd.factorize(ytr)[0]
            su1 = [symmetric_uncertainty(refine(np.zeros(len(yb), dtype=np.int64),
                   Xd[:, j]), yb) for j in range(m)]
            sel_h = sel_nog = [int(np.argmax(su1))]
            rec = []
        audit.append(dict(dataset=abbr, fold=fold, red_gd=sel_h,
                          red_nogd=sel_nog, same_reduct=(sel_h == sel_nog),
                          rounds=rec))
        print(f"  {abbr} fold {fold+1}/10 done", flush=True)
    return audit

if __name__ == "__main__":
    allrec = []
    t0 = time.perf_counter()
    for abbr in ORDER:
        print("auditing:", abbr, flush=True)
        try:
            allrec += audit_dataset(abbr)
        except Exception:
            import traceback; traceback.print_exc()
            print(f"!! {abbr} FAILED -- continuing", flush=True)
        json.dump(allrec, open("tie_audit.json", "w"))   # checkpoint
    print(f"\nDone: {len(allrec)} folds -> tie_audit.json "
          f"({time.perf_counter()-t0:.1f}s)")