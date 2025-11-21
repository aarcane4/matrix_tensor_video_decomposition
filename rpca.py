# RPCA
import numpy as np
from numpy.linalg import svd

def fro_norm(X):
    return np.sqrt(np.sum(X**2))

def singular_value_thresholding(X, tau):
    U, S, Vt = svd(X, full_matrices=False)
    S_th = np.maximum(S - tau, 0)
    return U @ np.diag(S_th) @ Vt

def soft_thresholding(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

def rpca(D, lam=None, mu=None, tol=1e-7, max_iter=300):
    D = D.astype(np.float64)
    m, n = D.shape

    if lam is None:
        lam = 1 / np.sqrt(max(m, n))
    if mu is None:
        mu = (m * n) / (4 * np.sum(np.abs(D)) + 1e-8)

    L = np.zeros_like(D)
    S = np.zeros_like(D)
    Y = np.zeros_like(D)

    for _ in range(max_iter):
        L = singular_value_thresholding(D - S + Y / mu, 1 / mu)
        S = soft_thresholding(D - L + Y / mu, lam / mu)

        R = D - L - S
        Y += mu * R

        if fro_norm(R) / fro_norm(D) < tol:
            break

    return L, S
