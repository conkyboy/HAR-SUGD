# -*- coding: utf-8 -*-
"""Unified loader. Every dataset -> (X DataFrame, y Series, kinds list).
CSVs read with header; .mat via scipy keys X/Y. '?' treated as missing
everywhere EXCEPT Mus (stalk-root '?' is an ordinary value by design).
Column dtypes decide kinds: 'num' / 'sym'."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.io import loadmat

DATA = Path(__file__).parent / "data"
FILES = {"Hil": "Hil.csv", "Sob": "Sob.csv", "Ger": "Ger.csv", "App": "App.csv",
         "Fla": "Fla.csv", "Spe": "Spe.csv", "Win": "Win.csv", "Wpb": "Wpb.csv",
         "Obs": "Obs.csv", "Dar": "Dar.csv", "Mus": "Mus.csv", "ORL40": "ORL40.csv",
         "Yal": "Yal.mat", "War": "War.mat", "Lun": "Lun.mat", "Pro": "Pro.mat",
         "Leu": "Leu.mat", "ORaw": "ORaw.mat"}

def load_dataset(abbr):
    f = DATA / FILES[abbr]
    if f.suffix == ".csv":
        df = pd.read_csv(f)
    else:
        m = loadmat(f)
        X, yv = np.asarray(m["X"], dtype=float), np.asarray(m["Y"]).ravel()
        df = pd.DataFrame(X, columns=[f"f{i+1}" for i in range(X.shape[1])])
        df["label"] = yv
    if abbr != "Mus":                       # '?' -> NaN (Mus keeps it as a value)
        df = df.replace("?", np.nan)
        for c in df.columns[:-1]:           # numeric strings -> numbers, if any
            try:
                df[c] = pd.to_numeric(df[c])
            except (ValueError, TypeError):
                pass
    y, X = df.iloc[:, -1], df.iloc[:, :-1]
    if abbr == "Leu":
        assert X.shape[1] == 7070, f"Leu features = {X.shape[1]}, expected 7070"
    kinds = []
    for c in X.columns:                     # value-based: dtype-API-proof
        try:
            pd.to_numeric(X[c]); kinds.append("num")
        except (ValueError, TypeError):
            kinds.append("sym")
    return X.reset_index(drop=True), y.reset_index(drop=True), kinds

def column_codes(df, kinds, n_bins=10):
    """Shared preprocessing (fit on the frame you pass in!).
    numeric -> equal-frequency bins (<= n_bins); symbolic -> integer codes.
    Returns (Xd int64 for SU-based selectors, Xnat float64 with numeric
    min-max [0,1] and symbolic codes for FSRMI/FDIWFS)."""
    n, m = df.shape
    Xd = np.empty((n, m), dtype=np.int64); Xnat = np.empty((n, m))
    for j, c in enumerate(df.columns):
        v = df[c].values
        if kinds[j] == "num":
            v = v.astype(float)
            edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1]))
            Xd[:, j] = np.searchsorted(edges, v, side="right")
            lo, hi = v.min(), v.max()
            Xnat[:, j] = (v - lo) / (hi - lo) if hi > lo else 0.0
        else:
            cats = pd.unique(pd.Series(v)); mp = {t: i for i, t in enumerate(cats)}
            codes = np.array([mp.get(t, len(cats)) for t in v], dtype=np.int64)
            Xd[:, j] = codes; Xnat[:, j] = codes
    return Xd, Xnat