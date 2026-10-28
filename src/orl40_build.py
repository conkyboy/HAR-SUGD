# -*- coding: utf-8 -*-
"""Assemble ORL40.csv from the AT&T orl_faces library (s1..s40 / 1..10.pgm).
Each 92x112 P5 pgm is mean-pooled to 32x32 = 1024 features; label =
person index 1..40. Output: data/ORL40.csv (400 x 1025, with header).
Pure numpy -- no Pillow needed."""
import numpy as np, pandas as pd
from pathlib import Path

SIZE = 32
ROOT = Path(__file__).parent / "data" / "orl_faces"
OUT = Path(__file__).parent / "data" / "ORL40.csv"

def read_pgm(p):
    with open(p, "rb") as f:
        assert f.readline().strip() == b"P5"
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        assert f.readline().strip() == b"255"
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w).astype(np.float64)

def block_mean(img, f=SIZE):
    """Area-average (mean pooling) HxW -> fxf, uneven blocks included."""
    H, W = img.shape
    r = (np.arange(H) * f) // H                 # target row of each source row
    c = (np.arange(W) * f) // W
    ri = r[:, None].repeat(W, 1).ravel(); ci = np.tile(c, H)
    out = np.zeros((f, f)); cnt = np.zeros((f, f))
    np.add.at(out, (ri, ci), img.ravel()); np.add.at(cnt, (ri, ci), 1.0)
    return out / cnt

rows = []
for s in range(1, 41):
    for n in range(1, 11):
        rows.append(list(block_mean(read_pgm(ROOT / f"s{s}" / f"{n}.pgm")).ravel()) + [s])
df = pd.DataFrame(rows, columns=[f"f{i+1}" for i in range(SIZE*SIZE)] + ["label"])
df.to_csv(OUT, index=False)
print(f"ORL40.csv: {df.shape} (expect (400, 1025)), classes: {df.label.nunique()}")