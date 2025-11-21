# metrics
import numpy as np

def fro_norm(X):
    return np.sqrt(np.sum(X**2))

def snr_db(true, est):
    true = true.astype(np.float64)
    est = est.astype(np.float64)

    err = fro_norm(true - est)
    if err == 0:
        err = 1e-12

    return 20 * np.log10(fro_norm(true) / err)

