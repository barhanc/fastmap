import numpy as np
import time

import fastmap.bfcm
import fastmap.proto

import mapel.elections.distances.cppdistances as dist

if __name__ == "__main__":
    n_votes, n_cands = 30, 9
    V1 = np.array([np.random.permutation(n_cands) for _ in range(n_votes)])
    V2 = np.array([np.random.permutation(n_cands) for _ in range(n_votes)])

    s = time.perf_counter()
    D = np.abs(np.subtract.outer(V1.argsort(), V2.argsort()), dtype=np.int32).swapaxes(1, 2)
    res = fastmap.bfcm.bfcm(D)
    e = time.perf_counter()
    print(f"C    : {res}, {e-s:.4f}s")

    s = time.perf_counter()
    res = dist.speard(V1, V2)
    e = time.perf_counter()
    print(f"C++  : {res}, {e-s:.4f}s")

    s = time.perf_counter()
    D = np.abs(np.subtract.outer(V1.argsort(), V2.argsort())).swapaxes(1, 2)
    res = fastmap.proto.bfcm(V1, V2, D)
    e = time.perf_counter()
    print(f"Numpy: {res}, {e-s:.4f}s")
