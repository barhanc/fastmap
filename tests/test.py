import numpy as np
import time
import fastmap.bfcm
import fastmap.isomorphic

from mapel.elections.distances import cppdistances as dist

if __name__ == "__main__":
    nv, nc = 5, 8
    print(f"Prob. size : nv={nv}, nc={nc}")

    X = np.array([np.random.permutation(nc) for _ in range(nv)])
    Y = np.array([np.random.permutation(nc) for _ in range(nv)])

    s = time.perf_counter()
    res = dist.speard(X, Y)
    e = time.perf_counter()
    t1 = e - s
    print(f"Target(C++): res={res}, t={e-s:.4f}s")

    s = time.perf_counter()
    pos_X, pos_Y = X.argsort(), Y.argsort()
    if nv < nc:
        pos_X, pos_Y = pos_X.T, pos_Y.T

    res = fastmap.bfcm.bfcm(pos_X.astype(np.int32), pos_Y.astype(np.int32))
    e = time.perf_counter()
    t2 = e - s
    print(f"Custom C   : res={res}, t={e-s:.4f}s")

    s = time.perf_counter()
    D = np.abs(np.subtract.outer(X.argsort(), Y.argsort())).swapaxes(1, 2)
    res = fastmap.isomorphic.bfcm(X, Y, D)
    e = time.perf_counter()
    t3 = e - s
    print(f"Numpy      : res={res}, t={e-s:.4f}s")

    print(f"{t1 / t2:4.2f}x")
