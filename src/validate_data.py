# -*- coding: utf-8 -*-
"""Final validation against the updated Table t2. ALL GREEN = ready."""
from load_dataset import load_dataset

NM_OK = {"Obs": 15, "Wpb": 4}   # documented missing counts (UCI)

EXPECTED = {  # abbr: (samples, features, classes)
    "Hil": (606, 100, 2), "Sob": (72, 19, 2), "Ger": (1000, 20, 2),
    "App": (106, 7, 2), "Fla": (1066, 11, 6), "Spe": (267, 44, 2),
    "Win": (178, 13, 3), "Wpb": (198, 33, 2), "Obs": (1075, 21, 4),
    "Dar": (174, 450, 2),
    "Yal": (165, 1024, 15), "ORL40": (400, 1024, 40), "War": (210, 2420, 10),
    "Lun": (203, 3312, 5), "Pro": (102, 5966, 2), "Leu": (72, 7070, 2),
    "ORaw": (100, 10304, 10), "Mus": (8124, 22, 2)}

bad = 0
for abbr, (n, m, k) in EXPECTED.items():
    try:
        X, y, kinds = load_dataset(abbr)
    except Exception as e:
        print(f"[{abbr:5s}] LOAD FAILED: {e}")
        bad += 1
        continue
    issues = []
    if X.shape != (n, m):
        issues.append(f"shape {X.shape} != {(n, m)}")
    if y.nunique() != k:
        issues.append(f"classes {y.nunique()} != {k}")
    nm = int(X.isna().sum().sum())
    exp_nm = NM_OK.get(abbr, 0)
    if nm != exp_nm:
        issues.append(f"NaN {nm} != expected {exp_nm}")
    info = (f"{X.shape[0]}x{X.shape[1]}, {y.nunique()} cls, "
            f"{kinds.count('sym')} sym cols")
    if issues:
        print(f"[{abbr:5s}] ✗ " + "; ".join(issues) + f"  [{info}]")
        bad += 1
    else:
        print(f"[{abbr:5s}] ✓  {info}")
print("\nALL GREEN — ready to run." if bad == 0 else f"\n{bad} problem(s).")