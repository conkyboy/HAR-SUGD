# verify_ties.py -- 平手候选是否诱导相同的当前加细划分？
import numpy as np, pandas as pd
from load_dataset import load_dataset, column_codes
from harsugd_core import har_sugd, refine, blocks_from_columns

events = identical = 0
for abbr in ["Hil", "App", "Dar", "Spe", "Wpb"]:      # 有平手的5个数据集
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
        Red = []
        for r in rec:
            if r["n_tied_exact"] > 1:
                tied = [int(j) for j, v in r["sigs"].items() if v == r["best_sig"]]
                P = blocks_from_columns(Xd[:, Red]) if Red else np.zeros(len(ytr), dtype=np.int64)
                refs = [refine(P, Xd[:, j]) for j in tied]
                events += 1
                identical += all(np.array_equal(refs[0], q) for q in refs[1:])
            Red.append(r["chosen"])
print(f"tie events={events}, identical refinements={identical}")
