# CP
import tensorly as tl
from tensorly.decomposition import parafac

def cp_reconstruct(X, rank=5):
    fac = parafac(X, rank=rank)
    return tl.cp_to_tensor(fac)

