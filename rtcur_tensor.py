# RTCUR
import numpy as np
import tensorly as tl
from numpy.linalg import pinv

def fro_norm(X):
    return np.sqrt(np.sum(X**2))

def rtcur_reconstruct(X, ranks=(10,10,3)):
    X = X.astype(np.float64)
    norm = fro_norm(X)
    Xn = X / (norm + 1e-12)

    factors = []

    for mode, r in enumerate(ranks):
        X_un = tl.unfold(Xn, mode)

        e_rows = np.sum(X_un**2, axis=1)
        e_cols = np.sum(X_un**2, axis=0)

        row_idx = np.argsort(e_rows)[-r:]
        col_idx = np.argsort(e_cols)[-r:]

        C = X_un[:, col_idx]
        R = X_un[row_idx, :]
        U = X_un[np.ix_(row_idx, col_idx)]

        U_pinv = pinv(U)
        factor = C @ U_pinv
        factors.append(factor)

    core = Xn
    for mode, F in enumerate(factors):
        core = tl.tenalg.mode_dot(core, pinv(F), mode)

    L = core
    for mode, F in enumerate(factors):
        L = tl.tenalg.mode_dot(L, F, mode)

    return L * norm, None
