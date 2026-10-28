# -*- coding: utf-8 -*-
"""FDIWFS — Ma, Xu & Liu, Applied Intelligence 55:166 (2025),
DOI 10.1007/s10489-024-06026-4. Formulas verified: FH (4), joint (5), MI (7),
CMI (8), interaction (10), FIF (14; denominator FH(fk,fs,y)), Rel (15),
J (16), weight update (17), Alg. 1 with threshold delta, sigma = per-feature
std, crisp relation for symbolic features. FIX vs previous delivery: the
decision relation is now ALWAYS crisp class equality (labels are
categorical; a Gaussian kernel on numeric label codes was a bug).

Input: X float64 (n, m), symbolic columns pre-encoded, kinds as in FSRMI."""
import numpy as np

BLK = 512

class _Rel:
    __slots__ = ("col", "symbolic", "sigma")
    def __init__(self, col, symbolic=False):
        self.col = np.asarray(col)
        self.symbolic = symbolic or self.col.dtype.kind in "USO"
        self.sigma = None if self.symbolic else float(np.std(self.col))
    def block(self, i0, i1):
        c = self.col[i0:i1]
        if self.symbolic:
            return (c[:, None] == self.col[None, :]).astype(np.float64)
        if self.sigma == 0.0:
            return np.ones((i1 - i0, self.col.size))
        d2 = c[:, None] - self.col[None, :]
        d2 *= d2
        d2 /= (2.0 * self.sigma ** 2)
        return np.exp(-d2)

def _cards(rels, n, block=BLK):
    out = np.empty(n)
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        m = rels[0].block(i0, i1)
        for r in rels[1:]:
            np.minimum(m, r.block(i0, i1), out=m)
        out[i0:i1] = m.sum(axis=1)
    return out

def _FH(c, n):
    return float(-np.mean(np.log(c / n)))

class FDIWFS:
    def select(self, X, y, delta, kinds=None):
        X = np.asarray(X, dtype=float); y = np.asarray(y)
        n, m = X.shape
        kinds = list(kinds) if kinds is not None else ["num"] * m
        Ry = _Rel(y, symbolic=True)                    # crisp class relation
        R = [_Rel(X[:, j], kinds[j] == "sym") for j in range(m)]
        Hy = _FH(_cards([Ry], n), n)
        H = [_FH(_cards([R[j]], n), n) for j in range(m)]
        Hfy = [_FH(_cards([R[j], Ry], n), n) for j in range(m)]
        S = [int(np.argmax([H[j] + Hy - Hfy[j] for j in range(m)]))]
        Hsy = {S[0]: Hfy[S[0]]}
        pair = {}
        def cmi_and_fif(j, s):
            key = (j, s) if j < s else (s, j)
            if key not in pair:
                pair[key] = (_FH(_cards([R[j], R[s]], n), n),
                             _FH(_cards([R[j], R[s], Ry], n), n))
            Hjs, Hjsy = pair[key]
            cmi = Hjs + Hsy[s] - Hjsy - H[s]                         # eq (8)
            inter = ((Hjs + Hy - Hjsy) - (H[j] + Hy - Hfy[j])
                     - (H[s] + Hy - Hsy[s]))                         # eq (10)
            fif = 1.0 + inter / Hjsy if inter > 0 and Hjsy > 0 else 1.0  # eq (14)
            return cmi, fif
        w = np.ones(m)
        while len(S) < delta and len(S) < m:                         # Alg. 1
            s_new = S[-1]
            cand = [j for j in range(m) if j not in S]
            for j in cand:
                w[j] *= cmi_and_fif(j, s_new)[1]                     # eq (17)
            best_j, best_J = None, -np.inf
            for j in cand:
                rel = min(cmi_and_fif(j, s)[0] for s in S)           # eq (15)
                if w[j] * rel > best_J:                              # eq (16)
                    best_j, best_J = j, w[j] * rel
            S.append(best_j)
            Hsy[best_j] = Hfy[best_j]
        return S