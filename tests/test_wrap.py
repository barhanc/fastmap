import time
import numpy as np
from itertools import permutations

import fastmap.example
import fastmap.example.wrapper as mdist
import fastmap.isomorphic

from mapel.elections.distances import cppdistances as dist

if __name__ == "__main__":
    nv, nc = 3, 4
    print(f"Prob. size : nv={nv}, nc={nc}")

    print("Spearman")
    U = np.array([np.random.permutation(nc) for _ in range(nv)])
    V = np.array([np.random.permutation(nc) for _ in range(nv)])

    s = time.perf_counter()
    res = dist.speard(U, V)
    e = time.perf_counter()
    t1 = e - s
    print(f"Target(C++): res={res}, t={e-s:.4f}s")

    s = time.perf_counter()
    res = mdist.spearman(U, V)
    e = time.perf_counter()
    t2 = e - s
    print(f"Custom C   : res={res}, t={e-s:.4f}s")

    s = time.perf_counter()
    D = np.abs(np.subtract.outer(U.argsort(), V.argsort())).swapaxes(1, 2)
    res = fastmap.isomorphic.bfcm(U, V, D)
    e = time.perf_counter()
    t3 = e - s
    print(f"Numpy      : res={res}, t={e-s:.4f}s")

    print(f"{t1 / t2:4.2f}x")

    print("Hamming")

    U = np.array([np.random.randint(0, 2, size=nc) for _ in range(nv)])
    V = np.array([np.random.randint(0, 2, size=nc) for _ in range(nv)])

    res = mdist.hamming(U, V)
    print(res)
