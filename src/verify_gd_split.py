# verify_gd_split.py
import numpy as np, pandas as pd
from load_dataset import load_dataset, column_codes
from harsugd_core import har_sugd

diff = same = 0
for abbr in ["Hil", "App", "Dar", "Spe", "Wpb"]:
    X, y, kinds = load_dataset(abbr)
    fold_id = pd.read_csv(f"folds/{abbr}.csv", index_col=0).iloc[:, 0].values
    for k in range(10):
        tr = np.where(fold_id != k)[0]
        Xtr = X.iloc[tr].copy(); ytr = y.iloc[tr].values
        for c, k_ in zip(X.columns, kinds):
            if Xtr[c].isna().any():
                Xtr[c] = Xtr[c].fillna(Xtr[c].median() if k_ == "num" else Xtr[c].mode()[0])
        Xd, _ = column_codes(Xtr, kinds)
        rec = []
        har_sugd(Xd, ytr, log=rec, log_full=True)
        for r in rec:
            if r["n_tied_exact"] > 1:
                gdv = [r["gds"][j] for j in r["sigs"] if r["sigs"][j] == r["best_sig"]]
                diff += max(gdv) > min(gdv); same += max(gdv) == min(gdv)
print(f"GD增益不同: {diff}, GD增益也相等: {same} (合计应=76)")