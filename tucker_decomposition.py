# Tucker
import tensorly as tl
from tensorly.decomposition import tucker

def tucker_reconstruct(X, ranks=(10,10,3)):
    core, factors = tucker(X, rank=ranks)
    return tl.tucker_to_tensor((core, factors))

