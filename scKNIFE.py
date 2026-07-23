import time

s = time.perf_counter()

import numpy as np
from numpy.random import default_rng
#import matplotlib.pyplot as plt
#import seaborn as sns

import scipy.sparse as sp
import scipy.sparse.linalg as spla
#from scKNIFE_graph.src.scKNIFE_graph.config import PRO_DIR

print(f"{time.perf_counter() - s : .3f}", "import times")

# R_g maps genes to nodes in unified graph

def NMF(X: sp.csr_array, r: int, iters: int, 
        epsilon: float=1e-9, seed: tuple[int, int]=(10,11)):
    # generate matrices of 1's with dimension N*r, r*p
    #W = np.ones((len(X), r), dtype=float)
    #H = np.ones((r, len(X[0])), dtype=float)
    N, M = X.shape
    W = default_rng(seed[0]).random((N, r), dtype=float)
    H = default_rng(seed[1]).random((r, M), dtype=float)

    coo = X.tocoo()
    rows, cols, x_data = coo.row, coo.col, coo.data

    def build_Q():
        loop = time.perf_counter()

        wh = np.zeros(len(rows), dtype=float)
        for i in range(W.shape[1]):
            wh += W[rows, i] * H[i, cols]

        print(time.perf_counter() - loop, "loop")

        
        Q = sp.csr_matrix((x_data / (wh + epsilon), (rows, cols)), shape=(N,M))
        return Q

    for t in range(iters):
        Q = build_Q()
        # H sum broadcasted over rest of the rows
        W *= Q.dot(H.T) / (H.sum(axis=1)[np.newaxis, :] + epsilon)

        Q = build_Q()
        # must be sparse.dot(dense)
        H *= (Q.T.dot(W)).T / (W.sum(axis=0)[:, np.newaxis] + epsilon)

    return W, H