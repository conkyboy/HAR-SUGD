# -*- coding: utf-8 -*-
"""Final nested-CV driver. Protocol (as disclosed in the manuscript):
stratified 10-fold (KFold fallback when any class has <10 samples, e.g.
Fla); per training fold: impute NaN with train-fold median/mode (Obs) ->
HAR-SUGD/MRMR/JMIM on fold-fitted equal-frequency codes; FSRMI/FDIWFS on
native values (numeric min-max [0,1], symbolic codes, FSRMI Fisher l per
dataset); MRMR/JMIM/FDIWFS return exactly len(HAR-SUGD) features.
Classification on raw-valued selected columns (symbolic -> fold-fitted
one-hot), KNN(k=3) and CART; Raw baseline on all columns."""
import time, numpy as np, pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

from load_dataset import load_dataset, column_codes
from harsugd_core import har_sugd, symmetric_uncertainty, refine
from baselines import mrmr_select, jmim_select
from fsrmi import FSRMI
from fdiwfs import FDIWFS

SEED = 42
ORDER = ["Hil", "Sob", "Ger", "App", "Fla", "Spe", "Win", "Wpb", "Obs",
         "Dar",
         "Yal", "ORL40", "War", "Lun", "Pro", "Leu", "ORaw", "Mus"]
FSRMI_L = {"Pro": 50}                       # Fisher top-l; default 250

def make_cv(y, n_splits=10):
    counts = np.bincount(pd.factorize(y)[0])
    if counts.min() >= n_splits:
        return StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    return KFold(n_splits, shuffle=True, random_state=SEED)   # e.g. Fla

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

def friedman_nemenyi(acc):
    k, N = acc.shape[1], acc.shape[0]
    ranks = acc.rank(axis=1, ascending=False).mean(axis=0)
    chi2 = 12.0 * N / (k * (k + 1)) * ((ranks ** 2).sum() - k * (k + 1) ** 2 / 4.0)
    F = (N - 1) * chi2 / (N * (k - 1) - chi2)
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}[k]
    cd = q * np.sqrt(k * (k + 1) / (6.0 * N))
    return ranks, chi2, F, cd

def run_dataset(abbr):
    X, y, kinds = load_dataset(abbr)
    m = X.shape[1]
    recs = []
    # --- persist fold assignment (audit-proof, version-stable) ---
    from pathlib import Path as _P
    _P("folds").mkdir(exist_ok=True)
    _f = _P(f"folds/{abbr}.csv")
    if _f.exists():
        import pandas as _pd
        fold_id = _pd.read_csv(_f, index_col=0).iloc[:, 0].values
        splits = [(np.where(fold_id != k)[0], np.where(fold_id == k)[0])
                  for k in range(10)]
    else:
        splits = list(make_cv(y).split(X, y))
        fold_id = np.zeros(len(y), dtype=int)
        for k, (_, te) in enumerate(splits):
            fold_id[te] = k
        pd.DataFrame({"fold": fold_id}).to_csv(_f)
    for fold, (tr, te) in enumerate(splits):
        Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
        ytr, yte = y.iloc[tr].values, y.iloc[te].values
        for c, k_ in zip(X.columns, kinds):            # fold-internal impute
            if Xtr[c].isna().any() or Xte[c].isna().any():
                fill = Xtr[c].median() if k_ == "num" else Xtr[c].mode()[0]
                Xtr[c] = Xtr[c].fillna(fill); Xte[c] = Xte[c].fillna(fill)
        Xd, Xnat = column_codes(Xtr, kinds)
        t0 = time.perf_counter(); sel_h = har_sugd(Xd, ytr); t_h = time.perf_counter() - t0
        if not sel_h:                                  # guard: empty reduct
            yb = pd.factorize(ytr)[0]
            su1 = [symmetric_uncertainty(refine(np.zeros(len(yb), dtype=np.int64),
                   Xd[:, j]), yb) for j in range(m)]
            sel_h = [int(np.argmax(su1))]
        k_sel = len(sel_h)
        t0 = time.perf_counter(); sel_m = mrmr_select(Xd, ytr, k_sel); t_m = time.perf_counter() - t0
        t0 = time.perf_counter(); sel_j = jmim_select(Xd, ytr, k_sel); t_j = time.perf_counter() - t0
        t0 = time.perf_counter()
        sel_f = FSRMI(fisher_top=FSRMI_L.get(abbr, 250)).select(Xnat, ytr, kinds=kinds)
        t_f = time.perf_counter() - t0
        t0 = time.perf_counter()
        sel_d = FDIWFS().select(Xnat, ytr, max(k_sel, 1), kinds=kinds)
        t_d = time.perf_counter() - t0
        sels = {"HAR-SUGD": (sel_h, t_h), "MRMR": (sel_m, t_m), "JMIM": (sel_j, t_j),
                "FSRMI": (sel_f, t_f), "FDIWFS": (sel_d, t_d)}
        for nm in list(sels):                     # last resort: never 0 columns
            if not sels[nm][0]:
                yb = pd.factorize(ytr)[0]
                su1 = [symmetric_uncertainty(refine(np.zeros(len(yb), dtype=np.int64),
                       Xd[:, j]), yb) for j in range(m)]
                sels[nm] = ([int(np.argmax(su1))], sels[nm][1])
        for clf_name, mk in [("KNN", lambda: KNeighborsClassifier(n_neighbors=3)),
                             ("CART", lambda: DecisionTreeClassifier(random_state=SEED))]:
            Atr, Ate = clf_matrices(Xtr, Xte, kinds, list(range(m)))
            recs.append(dict(abbr=abbr, fold=fold, clf=clf_name, method="Raw",
                             acc=mk().fit(Atr, ytr).score(Ate, yte), n_sel=m, t=0.0))
            for name, (sel, t) in sels.items():
                Atr, Ate = clf_matrices(Xtr, Xte, kinds, sel)
                recs.append(dict(abbr=abbr, fold=fold, clf=clf_name, method=name,
                                 acc=mk().fit(Atr, ytr).score(Ate, yte),
                                 n_sel=len(sel), t=t))
        print(f"  {abbr} fold {fold+1}/10 done", flush=True)
    return recs

if __name__ == "__main__":
    from pathlib import Path as _P
    if _P("results_raw.csv").exists():            # resume from checkpoint
        prev = pd.read_csv("results_raw.csv")
        allrec, done = prev.to_dict("records"), set(prev.abbr.unique())
        print(f"resume: skipping {len(done)} finished dataset(s)", flush=True)
    else:
        allrec, done = [], set()
    for abbr in ORDER:
        if abbr in done:
            print("skip (done):", abbr, flush=True)
            continue
        print("running:", abbr, flush=True)
        try:
            allrec += run_dataset(abbr)
        except Exception:
            import traceback; traceback.print_exc()
            print(f"!! {abbr} FAILED -- continuing", flush=True)
        pd.DataFrame(allrec).to_csv("results_raw.csv", index=False)   # checkpoint
    df = pd.DataFrame(allrec)
    summ = df.groupby(["clf", "abbr", "method"]).agg(
        acc=("acc", "mean"), std=("acc", "std"),
        n_sel=("n_sel", "mean"), t=("t", "sum")).reset_index()
    summ.to_csv("results_summary.csv", index=False)
    for clf in ["KNN", "CART"]:
        acc = summ[summ.clf == clf].pivot(index="abbr", columns="method", values="acc")
        acc = acc.reindex(ORDER)[["Raw", "MRMR", "JMIM", "FSRMI", "FDIWFS", "HAR-SUGD"]]
        print(f"\n===== {clf}: mean accuracy =====\n{acc.round(2)}")
        print(f"----- {clf}: paired t-tests vs HAR-SUGD -----")
        for mth in ["MRMR", "JMIM", "FSRMI", "FDIWFS"]:
            t, p = ttest_rel(acc["HAR-SUGD"], acc[mth])
            print(f"  vs {mth}: t={t:.3f}, p={p:.4f}")
        ranks, chi2, F, cd = friedman_nemenyi(acc[["MRMR", "JMIM", "FSRMI", "FDIWFS", "HAR-SUGD"]])
        print(f"Friedman: chi2={chi2:.4f}, F={F:.4f}, CD={cd:.4f}")
        print(f"mean ranks:\n{ranks.round(3)}")