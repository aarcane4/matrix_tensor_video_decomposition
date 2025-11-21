# CUR
import numpy as np

def cur_decomposition(D, k=10):
    m, n = D.shape

    col_energy = np.sum(D**2, axis=0)
    row_energy = np.sum(D**2, axis=1)

    col_idx = np.argsort(col_energy)[-k:]
    row_idx = np.argsort(row_energy)[-k:]

    C = D[:, col_idx]
    R = D[row_idx, :]
    U = D[np.ix_(row_idx, col_idx)]

    U_pinv = np.linalg.pinv(U)
    L = C @ U_pinv @ R

    return L
