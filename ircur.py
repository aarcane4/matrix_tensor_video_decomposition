# IRCUR
import numpy as np
from numpy.linalg import svd

def truncated_svd(X, k):
    U, S, Vt = svd(X, full_matrices=False)
    return U[:, :k], S[:k], Vt[:k, :]

def hard_threshold(X, tau):
    return np.where(np.abs(X) > tau, X, 0)

def fro_norm(X):
    return np.sqrt(np.sum(X**2))

def ircur(D, r=10, z0=3, gamma=0.95, max_iter=8):
    Dn = D / (np.max(np.abs(D)) + 1e-8)

    L = np.zeros_like(Dn)
    S = np.zeros_like(Dn)
    z = z0

    for _ in range(max_iter):
        S = hard_threshold(Dn - L, z)
        z *= gamma

        R = Dn - S

        row_e = np.sum(R**2, axis=1)
        col_e = np.sum(R**2, axis=0)

        rows = np.argsort(row_e)[-r:]
        cols = np.argsort(col_e)[-r:]

        C = R[:, cols]
        R_sub = R[rows, :]
        U = R[np.ix_(rows, cols)]

        Uu, Ss, Vv = truncated_svd(U, r)
        U_pinv = Vv.T @ np.diag(1/(Ss + 1e-12)) @ Uu.T

        L = C @ U_pinv @ R_sub

    L *= np.max(np.abs(D))
    S *= np.max(np.abs(D))
    return L, S
