# -*- coding: utf-8 -*-
"""MRMR (MID criterion) and JMIM selectors on discretized codes.
Both take Xd (n x m int bin codes), y (any labels), k (subset size),
return a list of k column indices. MI: plugin estimator on the
empirical joint (natural log; invariant for ranking)."""
import numpy as np

def _codes(v):
    _, c = np.unique(np.asarray(v), return_inverse=True)
    return c.astype(np.int64)

def _mi(a, b):
    n = len(a)
    kb = int(b.max()) + 1
    cab = np.bincount(a * kb + b, minlength=(int(a.max()) + 1) * kb)
    cab = cab.astype(float).reshape(-1, kb)
    ca, cb = cab.sum(1), cab.sum(0)
    nz = cab > 0
    pab = cab / n
    outer = (ca / n)[:, None] * (cb / n)[None, :]
    return float((pab[nz] * np.log(pab[nz] / outer[nz])).sum())

def _refine(a, b):
    _, inv = np.unique(a.astype(np.int64) * (int(b.max()) + 1) + b.astype(np.int64),
                       return_inverse=True)
    return inv.astype(np.int64)

def _pair_mi(cols, i, j, cache):
    key = (i, j) if i < j else (j, i)
    if key not in cache:
        cache[key] = _mi(cols[i], cols[j])
    return cache[key]

def mrmr_select(Xd, y, k):
    Xd = np.asarray(Xd); yb = _codes(y)
    m = Xd.shape[1]; k = min(int(k), m)
    cols = [_codes(Xd[:, j]) for j in range(m)]
    miy = np.array([_mi(c, yb) for c in cols])
    S = [int(np.argmax(miy))]
    cache = {}
    while len(S) < k:
        best_v, best_j = -np.inf, None
        for j in range(m):
            if j in S: continue
            red = sum(_pair_mi(cols, j, s, cache) for s in S) / len(S)
            if miy[j] - red > best_v:
                best_v, best_j = miy[j] - red, j
        S.append(best_j)
    return S

def jmim_select(Xd, y, k):
    Xd = np.asarray(Xd); yb = _codes(y)
    m = Xd.shape[1]; k = min(int(k), m)
    cols = [_codes(Xd[:, j]) for j in range(m)]
    miy = np.array([_mi(c, yb) for c in cols])
    S = [int(np.argmax(miy))]
    cache = {}
    while len(S) < k:
        best_v, best_j = -np.inf, None
        for j in range(m):
            if j in S: continue
            # I(f;y|s) = I((f,s);y) - I(s;y), cached per (j,s)
            v = None
            for s in S:
                if (j, s) not in cache:
                    cache[(j, s)] = _mi(_refine(cols[j], cols[s]), yb) - miy[s]
                cmi = cache[(j, s)]
                v = cmi if v is None else min(v, cmi)
            if v > best_v:
                best_v, best_j = v, j
        S.append(best_j)
    return S

def fcbf_select(Xd, y, delta=0.5):
    """FCBF (Yu & Liu 2004) on integer codes; self-determined subset size."""
    import numpy as np
    from harsugd_core import symmetric_uncertainty, _codes
    Xd = np.asarray(Xd); yb = _codes(y)
    m = Xd.shape[1]
    su_d = np.array([symmetric_uncertainty(Xd[:, j], yb) for j in range(m)])
    S = [int(j) for j in range(m) if su_d[j] > delta]     # pre-dominance 过滤
    if not S:
        return [int(np.argmax(su_d))]                      # 空预过滤兜底
    S.sort(key=lambda j: -su_d[j])                         # 按 SU(·,d) 降序
    out = []
    while S:
        f = S.pop(0); out.append(f)
        S = [g for g in S                                  # f 支配 g 则删 g
             if symmetric_uncertainty(Xd[:, f], Xd[:, g]) < su_d[g]]
    return out