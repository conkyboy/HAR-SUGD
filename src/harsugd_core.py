# -*- coding: utf-8 -*-
"""HAR-SUGD core -- calibrated line by line against Algorithm 1 (Section 4).

Locked semantics (hand-verified on the Table I toy system):
  refine(P, Q)                 common refinement of two partitions
  blocks_from_columns(M)       joint partition of all columns of M
  symmetric_uncertainty(P, y)  SU = 2 I(P;y) / (H(P) + H(y))
  knowledge_granularity(P)     GD(P) = sum |P_i|^2 / |U|^2  (Def. 2.2)

Algorithm 1: forward greedy; each round selects argmax
sig_su(a) = SU(Red+{a}, d) - SU(Red, d), ties broken by the larger
granularity gain GD(a|Red) = GD(Red) - GD(Red+{a}); stops when no
candidate exceeds eps (1e-9 per the manuscript). No backward phase.

Unit test anchors (hand-verified; see chat): SU=[0.7934, 0.7640, 0.7294],
GD=[0.34, 0.52, 0.30], Red=[0, 1], SU(Red,d)=0.9064, SU(A,d)=0.8397.
The manuscript's Remark quotes 0.6706 / -0.1228 for sig_su(a3|{a1});
that contradicts its own Theorem 4.1 and Section-3 Example -- ledger item."""
import numpy as np

def _codes(v):
    _, c = np.unique(np.asarray(v), return_inverse=True)
    return c.astype(np.int64)

def refine(P, Q):
    a, b = _codes(P), _codes(Q)
    _, inv = np.unique(a * (int(b.max()) + 1) + b, return_inverse=True)
    return inv.astype(np.int64)

def blocks_from_columns(M):
    M = np.asarray(M)
    if M.ndim == 1:
        return _codes(M)
    if M.shape[1] == 0:
        return np.zeros(M.shape[0], dtype=np.int64)
    P = _codes(M[:, 0])
    for j in range(1, M.shape[1]):
        P = refine(P, M[:, j])
    return P

def _entropy(c):
    p = np.bincount(c) / len(c)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def symmetric_uncertainty(P, y):
    a, b = _codes(P), _codes(y)
    Ha, Hb = _entropy(a), _entropy(b)
    if Ha + Hb <= 0:
        return 0.0
    Hab = _entropy(refine(a, b))
    return 2.0 * (Ha + Hb - Hab) / (Ha + Hb)

def knowledge_granularity(P):
    """GD(P) = sum_i |P_i|^2 / |U|^2 (Def. 2.2). Smaller = finer."""
    c = _codes(P)
    cnt = np.bincount(c).astype(float)
    return float((cnt ** 2).sum() / (len(c) ** 2))

def har_sugd(Xd, y, eps=1e-9, tie_break="gd", log=None, log_full=False):
    """Algorithm 1. Xd: (n, m) integer codes; y: labels. -> column list.

    tie_break="gd"   : Algorithm 1 as written -- exact sig ties broken by
                       the larger granularity gain GD(a|Red).
    tie_break="none" : W-R1 ablation -- ties resolved by column order,
                       granularity never consulted.

    log: optional list; one record per round:
         {round, n_candidates, best_sig, n_tied_exact,
          tie_changed_choice, chosen} (+ full sigs/gds if log_full).
         Keep log_full=False on 7k-10k-column datasets unless needed.
    """
    Xd = np.asarray(Xd); yb = _codes(y)
    n, m = Xd.shape
    cols = [Xd[:, j] for j in range(m)]
    P = np.zeros(n, dtype=np.int64)     # Red = {} -> one-block partition
    su_cur = 0.0                        # SU({}, d) = 0
    gd_cur = 1.0                        # GD({})  = 1
    Red, in_R = [], np.zeros(m, dtype=bool)
    t = 0
    while True:                         # Algorithm 1, lines 2-12
        sigs, gds = {}, {}
        for j in range(m):
            if in_R[j]:
                continue
            Pj = refine(P, cols[j])
            sigs[j] = symmetric_uncertainty(Pj, yb) - su_cur        # line 5
            gds[j]  = gd_cur - knowledge_granularity(Pj)            # line 6
        if not sigs or max(sigs.values()) <= eps:                   # line 11
            break
        best_sig = max(sigs.values())
        tied = [j for j in sigs if sigs[j] == best_sig]             # exact ties
        best_j = (max(tied, key=lambda j: gds[j]) if tie_break == "gd"
                  else tied[0])                                     # lines 7-8
        if log is not None:
            rec = {"round": t, "n_candidates": len(sigs),
                   "best_sig": float(best_sig), "n_tied_exact": len(tied),
                   "tie_changed_choice": bool(len(tied) > 1 and best_j != tied[0]),
                   "chosen": int(best_j)}
            if log_full:
                rec["sigs"] = {int(j): float(v) for j, v in sigs.items()}
                rec["gds"]  = {int(j): float(v) for j, v in gds.items()}
            log.append(rec)
        Red.append(int(best_j)); in_R[best_j] = True
        P = refine(P, cols[best_j])
        su_cur = symmetric_uncertainty(P, yb)
        gd_cur = knowledge_granularity(P)
        t += 1
    return Red

if __name__ == "__main__":
    # Table I (tab17): columns a1, a2, a3, decision d LAST.
    TABLE1 = np.array([
        [2, 1, 3, 1],
        [3, 2, 1, 2],
        [2, 1, 3, 1],
        [2, 2, 3, 2],
        [1, 1, 4, 3],
        [1, 1, 2, 3],
        [3, 2, 1, 2],
        [1, 1, 4, 3],
        [2, 1, 3, 1],
        [3, 2, 1, 2]])
    Xd, d = TABLE1[:, :-1], TABLE1[:, -1]
    su1 = [round(symmetric_uncertainty(Xd[:, j], d), 4) for j in range(3)]
    gd1 = [round(knowledge_granularity(Xd[:, j]), 4) for j in range(3)]
    red = har_sugd(Xd, d)
    su_red = round(symmetric_uncertainty(blocks_from_columns(Xd[:, red]), d), 4)
    su_all = round(symmetric_uncertainty(blocks_from_columns(Xd), d), 4)
    print("SU(a_i, d) :", su1)
    print("GD(a_i)    :", gd1)
    print("Red        :", red, "(selection order)")

    def _ok(name, got, exp, tol=1e-3):
        good = len(got) == len(exp) and all(abs(g - e) <= tol
                                            for g, e in zip(got, exp))
        print(f"  {name}: {'PASS' if good else 'FAIL'} (expected {exp})")
        return good
    n_ok = sum([
        _ok("SU(a_i,d)", su1, [0.7934, 0.7640, 0.7294]),
        _ok("GD(a_i)  ", gd1, [0.34, 0.52, 0.30]),
        _ok("Red      ", sorted(red), [0, 1]),
        _ok("SU(Red,d)", [su_red], [0.9064]),
        _ok("SU(A,d)  ", [su_all], [0.8397])])
    print("5/5 anchors passed -- HAR-SUGD LOCKED." if n_ok == 5
          else f"{n_ok}/5 anchors passed -- send me this output.")