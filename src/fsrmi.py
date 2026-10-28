# -*- coding: utf-8 -*-
"""FSRMI — Qu, Xu, Han & Xu, Applied Intelligence 53(14):17727-17746 (2023),
DOI 10.1007/s10489-022-04398-z. All transcription points verified against
the PDF: B (eq 12); Delta without 1/|S| normalization; neighborhood <= Ad
(eq 14); Alg. 1 forward argmax Router>0 + single-pass backward Rinner<=0;
Fisher prefilter l per dataset; min-max [0,1] input (our disclosed choice).

Input: X float64 (n, m), symbolic columns pre-encoded as integer codes,
kinds[j] in {'num','sym'}. Distance: d2 = sum_num (xik-xjk)^2 + sum_sym
mism; Delta = sqrt(d2) per the paper's Delta_F. Hamming extension for
symbolic data is ours (Qu et al. evaluated continuous datasets only)."""
import numpy as np

BLK = 512

class FSRMI:
    def __init__(self, fisher_threshold=1000, fisher_top=250):
        self.fisher_threshold = fisher_threshold
        self.fisher_top = fisher_top

    @staticmethod
    def _fisher(X, y):
        mu = X.mean(0)
        num = np.zeros(X.shape[1]); den = np.zeros(X.shape[1]) + 1e-12
        for c in np.unique(y):
            Xc = X[y == c]
            num += len(Xc) * (Xc.mean(0) - mu) ** 2
            den += len(Xc) * Xc.var(0)
        return num / den

    def select(self, X, y, k=None, kinds=None):
        X = np.asarray(X, dtype=float); y = np.asarray(y)
        n, m = X.shape
        kinds = list(kinds) if kinds is not None else ["num"] * m
        Xc = X.copy()
        for j in range(m):                       # min-max numeric columns
            if kinds[j] == "num":
                lo, hi = Xc[:, j].min(), Xc[:, j].max()
                if hi > lo:
                    Xc[:, j] = (Xc[:, j] - lo) / (hi - lo)
        keep = np.arange(m)
        if m > self.fisher_threshold:            # numeric-only path (ASU sets)
            keep = np.argsort(-self._fisher(Xc, y))[:self.fisher_top]
            Xc = Xc[:, keep]; kinds = [kinds[j] for j in keep]; m = Xc.shape[1]

        ycode = np.unique(y, return_inverse=True)[1]
        cls_size = np.bincount(ycode)[ycode]
        same = ycode[:, None] == ycode[None, :]

        def feat_block(j, i0, i1):
            col = Xc[:, j]
            if kinds[j] == "sym":
                return (col[i0:i1, None] != col[None, :]).astype(np.float64)
            d = col[i0:i1, None] - col[None, :]
            return d * d

        # ---- boundaries B (eq 12), adaptive radius Ad (eq 13) ----
        d_friend = np.full(n, np.inf); d_enemy = np.full(n, np.inf)
        for i0 in range(0, n, BLK):
            i1 = min(i0 + BLK, n)
            d2 = np.zeros((i1 - i0, n))
            for j in range(m):
                d2 += feat_block(j, i0, i1)
            d2[np.arange(i1 - i0), np.arange(i0, i1)] = np.inf   # exclude self
            dm = np.sqrt(d2)
            d_friend[i0:i1] = np.where(same[i0:i1], dm, np.inf).min(1)
            d_enemy[i0:i1] = np.where(~same[i0:i1], dm, np.inf).min(1)
        ok = np.isfinite(d_friend) & np.isfinite(d_enemy)
        B = np.where(ok, np.maximum(0.0, d_enemy - d_friend), 0.0)
        Ad = np.zeros(n)
        for c in np.unique(ycode):
            msk = ycode == c
            Ad[msk] = B[msk].mean()
        rad2 = Ad ** 2                           # d2 <= Ad^2 <=> Delta <= Ad

        # ---- RM(D;S) = gamma_S(D) * M(D;S) (Prop. 2) ----
        D2S = np.zeros((n, n))
        def rm_of(extra=None, minus=None):
            lb = 0; s_card = np.empty(n); s_same = np.empty(n)
            for i0 in range(0, n, BLK):
                i1 = min(i0 + BLK, n)
                d2 = D2S[i0:i1].copy()
                if extra is not None: d2 += feat_block(extra, i0, i1)
                if minus is not None: d2 -= feat_block(minus, i0, i1)
                nb = d2 <= rad2[i0:i1, None]                    # eq (14)
                card = nb.sum(1)
                sc = (nb & same[i0:i1]).sum(1)
                s_card[i0:i1] = card; s_same[i0:i1] = sc
                lb += int((sc == card).sum())                   # eq (18)
            gamma = 1.0 - lb / n                                # eq (21)
            M = -np.mean(np.log(cls_size * s_card / (n * s_same)))  # eq (26)
            return gamma * M
        def commit(j):
            for i0 in range(0, n, BLK):
                i1 = min(i0 + BLK, n)
                D2S[i0:i1] += feat_block(j, i0, i1)
        def rebuild(S):
            D2S[:] = 0.0
            for j in S: commit(j)

        # ---- forward: max-relevance subset (Alg. 1, steps 2-7) ----
        S = []
        while True:
            best_j, best_r = None, 0.0
            in_S = np.zeros(m, dtype=bool); in_S[S] = True
            rm_cur = rm_of()
            for j in range(m):
                if in_S[j]: continue
                r = rm_of(extra=j) - rm_cur                     # Router, eq (29)
                if r > best_r: best_j, best_r = j, r
            if best_j is None: break
            S.append(best_j); commit(best_j)
        if not S:                # degenerate guard: empty reduct unusable
            S = [int(np.argmax([rm_of(extra=j) for j in range(m)]))]
            commit(S[0])
        rm_S = rm_of()

        # ---- backward: single pass in selection order (steps 8-12) ----
        for j in list(S):
            if rm_of(minus=j) >= rm_S:                          # Rinner<=0, eq (28)
                S.remove(j); rebuild(S); rm_S = rm_of()
        if not S:                # backward emptied S (all Rinner<=0): top-1 by RM
            S = [int(np.argmax([rm_of(extra=j) for j in range(m)]))]
            rebuild(S)
        return [int(keep[j]) for j in S]
        