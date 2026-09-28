"""Méthodes de pondération : CRITIC, Entropie, AHP, BWM."""
import numpy as np
from scipy.optimize import linprog


def normalize(X, benefit):
    """Normalisation min-max (bénéfice / coût) -> [0,1]."""
    X = np.asarray(X, dtype=float)
    mn, mx = X.min(axis=0), X.max(axis=0)
    rng = np.where(mx - mn == 0, 1, mx - mn)
    N = np.where(benefit, (X - mn) / rng, (mx - X) / rng)
    return N


def critic(X, benefit):
    N = normalize(X, benefit)
    sigma = N.std(axis=0, ddof=1)
    R = np.nan_to_num(np.corrcoef(N, rowvar=False))
    C = sigma * (1 - np.abs(R)).sum(axis=0)
    return C / C.sum() if C.sum() else np.ones(len(C)) / len(C)


def entropy(X, benefit):
    N = normalize(X, benefit)
    S = N.sum(axis=0)
    P = N / np.where(S == 0, 1, S)
    m = X.shape[0]
    plogp = np.where(P > 0, P * np.log(np.where(P > 0, P, 1)), 0.0)
    E = -plogp.sum(axis=0) / np.log(m)          # E_j = -k sum p ln p
    n = X.shape[1]
    return (1 - E) / (n - E.sum())              # w_j = (1-E_j)/(n-sum E_k)


RI = {3: .58, 4: .90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.56}


def ahp(A):
    """Méthode approximative du cours : somme des colonnes s_j, division, moyenne des lignes = w,
    lambda_max = sum(s_j * w_j), CI = (lmax-n)/(n-1), CR = CI/RI. Retourne (poids, CR)."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    s = A.sum(axis=0)
    w = (A / s).mean(axis=1)
    lam = float((s * w).sum())
    CI = (lam - n) / (n - 1) if n > 1 else 0.0
    ri = RI.get(n, 1.56 if n > 10 else 0)
    CR = CI / ri if ri else 0.0
    return w, CR


def ahp_from_upper(M):
    """Construit la matrice réciproque à partir du triangle supérieur."""
    M = np.array(M, dtype=float)
    n = M.shape[0]
    A = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = M[i, j]
            A[j, i] = 1 / M[i, j]
    return A


BWM_CI = {1: 0, 2: 0.44, 3: 1.0, 4: 1.63, 5: 2.3, 6: 3.0, 7: 3.73, 8: 4.47, 9: 5.23}


def bwm(best, worst, b_to_o, o_to_w):
    """BWM linéaire. best/worst : indices ; b_to_o, o_to_w : vecteurs (1..9).
    Retourne (poids, xi, CR)."""
    n = len(b_to_o)
    # variables : w_0..w_{n-1}, xi
    c = np.zeros(n + 1); c[-1] = 1
    A, b = [], []
    for j in range(n):
        for s in (1, -1):
            row = np.zeros(n + 1)
            row[best] += s; row[j] -= s * b_to_o[j]; row[-1] = -1
            A.append(row); b.append(0)
            row = np.zeros(n + 1)
            row[j] += s; row[worst] -= s * o_to_w[j]; row[-1] = -1
            A.append(row); b.append(0)
    Aeq = [np.append(np.ones(n), 0)]
    res = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=[1],
                  bounds=[(0, None)] * (n + 1), method="highs")
    w = res.x[:n]
    xi = res.x[-1]
    return w, xi
