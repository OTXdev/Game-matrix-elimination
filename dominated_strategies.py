from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
_EPS = 1e-12

def is_row_dominated_by_mixed(u1, row_idx, other_rows):
   
    n_cols = u1.shape[1]
    n_mix  = len(other_rows)
    if n_mix == 0:
        return False, None

    c    = np.zeros(n_mix)
    A_ub = np.zeros((n_cols, n_mix))
    b_ub = np.zeros(n_cols)

    for j in range(n_cols):
        for k, r in enumerate(other_rows):
            A_ub[j, k] = -u1[r, j]
        b_ub[j] = -u1[row_idx, j] - _EPS

    A_eq = np.ones((1, n_mix))
    b_eq = np.array([1.0])

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0.0, 1.0)] * n_mix, method="highs")
    if res.success:
        p = np.clip(res.x, 0, 1)
        p /= p.sum()
        return True, p
    return False, None


def is_col_dominated_by_mixed(u2, col_idx, other_cols):
   
    n_rows = u2.shape[0]
    n_mix  = len(other_cols)
    if n_mix == 0:
        return False, None

    c    = np.zeros(n_mix)
    A_ub = np.zeros((n_rows, n_mix))
    b_ub = np.zeros(n_rows)

    for i in range(n_rows):
        for k, col in enumerate(other_cols):
            A_ub[i, k] = -u2[i, col]
        b_ub[i] = -u2[i, col_idx] - _EPS

    A_eq = np.ones((1, n_mix))
    b_eq = np.array([1.0])

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0.0, 1.0)] * n_mix, method="highs")
    if res.success:
        q = np.clip(res.x, 0, 1)
        q /= q.sum()
        return True, q
    return False, None


def iterated_elimination(payoffs, row_labels, col_labels, max_rounds=100, verbose=True):
   
    P        = payoffs.copy().astype(float)
    r_lab    = list(row_labels)
    c_lab    = list(col_labels)
    full_log = []

    if verbose:
        print("=" * 60)
        print("  IESDS — Deux joueurs, chacun maximise son gain")
        print("=" * 60)
        _print_matrix(P, r_lab, c_lab, "Matrice initiale")

    for round_num in range(max_rounds):
        prev_shape = P.shape[:2]
        round_log  = []

        # ---- Joueur 1 ----
        u1      = P[:, :, 0]
        changed = True
        while changed:
            changed = False
            for i in range(len(r_lab)):
                others = [k for k in range(len(r_lab)) if k != i]
                dominated, p = is_row_dominated_by_mixed(u1, i, others)
                if dominated:
                    mix_desc = {r_lab[others[k]]: round(float(p[k]), 4)
                                for k in range(len(others)) if p[k] > 1e-4}
                    entry = {"player": "Joueur 1",
                             "eliminated": r_lab[i],
                             "dominated_by": mix_desc}
                    round_log.append(entry)
                    if verbose:
                        mix_str = " + ".join(f"{v:.3f}·{k}" for k,v in mix_desc.items())
                        print(f"\n  [J1] Stratégie '{r_lab[i]}' éliminée")
                        print(f"       Dominée par: {mix_str}")
                    P     = np.delete(P, i, axis=0)
                    r_lab = [r_lab[k] for k in range(len(r_lab)) if k != i]
                    u1    = P[:, :, 0]
                    changed = True
                    break

        # ---- Joueur 2 ----
        u2      = P[:, :, 1]
        changed = True
        while changed:
            changed = False
            for j in range(len(c_lab)):
                others = [k for k in range(len(c_lab)) if k != j]
                dominated, q = is_col_dominated_by_mixed(u2, j, others)
                if dominated:
                    mix_desc = {c_lab[others[k]]: round(float(q[k]), 4)
                                for k in range(len(others)) if q[k] > 1e-4}
                    entry = {"player": "Joueur 2",
                             "eliminated": c_lab[j],
                             "dominated_by": mix_desc}
                    round_log.append(entry)
                    if verbose:
                        mix_str = " + ".join(f"{v:.3f}·{k}" for k,v in mix_desc.items())
                        print(f"\n  [J2] Stratégie '{c_lab[j]}' éliminée")
                        print(f"       Dominée par: {mix_str}")
                    P     = np.delete(P, j, axis=1)
                    c_lab = [c_lab[k] for k in range(len(c_lab)) if k != j]
                    u2    = P[:, :, 1]
                    changed = True
                    break

        full_log.extend(round_log)

        if P.shape[:2] == prev_shape:
            break

    if verbose:
        _print_matrix(P, r_lab, c_lab, "Matrice réduite finale")

    solved = (P.shape[0] == 1 and P.shape[1] == 1)
    if verbose:
        print("\n" + "=" * 60)
        if solved:
            print(f"  Solution unique: ({r_lab[0]}, {c_lab[0]})")
            print(f"  Gains: u1={P[0,0,0]:.4f}, u2={P[0,0,1]:.4f}")
        else:
            print(f"  Matrice {P.shape[0]}×{P.shape[1]} — analyse Nash nécessaire")
            print(f"  Joueur 1 : {r_lab}")
            print(f"  Joueur 2 : {c_lab}")
        print("=" * 60)

    return {
        "payoffs":         P,
        "u1":              P[:, :, 0],
        "u2":              P[:, :, 1],
        "row_labels":      r_lab,
        "col_labels":      c_lab,
        "elimination_log": full_log,
        "n_rounds":        round_num + 1,
        "solved":          solved,
    }


def _print_matrix(P, row_labels, col_labels, title="Matrice des gains"):
    col_w  = max(12, max((len(c) for c in col_labels), default=0) + 4)
    row_lw = max((len(r) for r in row_labels), default=0) + 2
    sep    = "  " + "-" * (row_lw + col_w * len(col_labels) + 2)
    print(f"\n  {title}")
    print(sep)
    print("  " + " " * row_lw + "".join(c.center(col_w) for c in col_labels))
    print(sep)
    for i, rl in enumerate(row_labels):
        row = rl.ljust(row_lw)
        for j in range(len(col_labels)):
            cell = f"({P[i,j,0]:.1f},{P[i,j,1]:.1f})"
            row += cell.center(col_w)
        print("  " + row)
    print(sep)


def print_matrix(P, row_labels, col_labels, title="Matrice des gains"):
    _print_matrix(P, row_labels, col_labels, title)


def demo():
    raw = [
        [(3,5),(2,0),(2,2)],
        [(5,2),(1,2),(2,1)],
        [(9,0),(1,5),(3,2)],
    ]
    P = np.array(raw, dtype=float)
    iterated_elimination(P, ["A","B","C"], ["L","M","R"], verbose=True)

if __name__ == "__main__":
    demo()