import numpy as np
import time

import fastmap.bfcm
import fastmap.proto

import mapel.elections.distances.cppdistances as dist

if __name__ == "__main__":
    n_votes, n_cands = 8, 8
    V1 = np.array([np.random.permutation(n_cands) for _ in range(n_votes)])
    V2 = np.array([np.random.permutation(n_cands) for _ in range(n_votes)])
    # print(V1.argsort())
    # print(V2.argsort())

    s = time.perf_counter()
    # Easy numpy calculation of Spearman distance tensor. But data passing between numpy and C for
    # large tensor D is kinda slow. We should try passing V1 and V2 and calculating D in pure C
    # D = np.abs(np.subtract.outer(V1.argsort(), V2.argsort()), dtype=np.int32).swapaxes(1, 2)
    res = fastmap.bfcm.bfcm(V1.argsort().astype(np.int32), V2.argsort().astype(np.int32))
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
