# -*- coding: utf-8 -*-
"""eps-invariance check (P2-6): on the same 180 training folds, are the
HAR-SUGD reducts identical for eps = 1e-3, 1e-6, 1e-12 vs the paper's
1e-9? Writes eps_check.json and prints a 3-line summary."""
import inspect, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

import harsugd_core
from load_dataset import load_dataset, column_codes
from harsugd_core import har_sugd

SEED = 42
ORDER = ["Hil", "Sob", "Ger", "App", "Fla", "Spe", "Win", "Wpb", "Obs",
         "Dar", "Yal", "ORL40", "War", "Lun", "Pro", "Leu", "ORaw", "Mus"]
EPS_LIST = [("1e-3", 1e-3), ("1e-6", 1e-6), ("1e-12", 1e-12)]

_params = inspect.signature(har_sugd).parameters
HAS_EPS_PARAM = "eps" in _params
HAS_EPS_GLOBAL = hasattr(harsugd_core, "EPS")

def hs(Xd, ytr, eps):
    if HAS_EPS_PARAM:
        return list(har_sugd(Xd, ytr, eps=eps))
    if HAS_EPS_GLOBAL:                   # module constant read at call time
        old = harsugd_core.EPS
        harsugd_core.EPS = eps
        try:
            return list(har_sugd(Xd, ytr))
        finally:
            harsugd_core.EPS = old
    raise RuntimeError("no eps interface")

def make_cv(y, n_splits=10):             # verbatim from main driver
    counts = np.bincount(pd.factorize(y)[0])
    if counts.min() >= n_splits:
        return StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    return KFold(n_splits, shuffle=True, random_state=SEED)

if __name__ == "__main__":
    print(f"[eps interface] parameter={HAS_EPS_PARAM}, "
          f"EPS global={HAS_EPS_GLOBAL}")
    if not (HAS_EPS_PARAM or HAS_EPS_GLOBAL):
        print("!! har_sugd 没有可用的 eps 接口，请把下面两行贴回来：")
        print("   signature:", inspect.signature(har_sugd))
        print("   module attrs:", [a for a in dir(harsugd_core) if a.isupper()])
        raise SystemExit(1)

    out = []
    for abbr in ORDER:
        print("eps-check:", abbr, flush=True)
        X, y, kinds = load_dataset(abbr)
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
            for c, k_ in zip(X.columns, kinds):     # same imputation
                if Xtr[c].isna().any():
                    fill = Xtr[c].median() if k_ == "num" else Xtr[c].mode()[0]
                    Xtr[c] = Xtr[c].fillna(fill)
            Xd, _ = column_codes(Xtr, kinds)
            base = hs(Xd, ytr, 1e-9)
            row = dict(abbr=abbr, fold=fold, base_len=len(base))
            for tag, e in EPS_LIST:
                row[f"same_{tag}"] = (hs(Xd, ytr, e) == base)
            out.append(row)
    json.dump(out, open("eps_check.json", "w"))

    tot = len(out)
    print()
    for tag, _ in EPS_LIST:
        n_diff = sum(1 for r in out if not r[f"same_{tag}"])
        print(f"eps={tag}: differing reducts {n_diff}/{tot}")