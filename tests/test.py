import numpy as np
import time
import fastmap.bfcm
import fastmap.proto

import mapel.elections.distances.cppdistances as dist
import mapel.elections.distances

if __name__ == "__main__":
    n_votes, n_cands = 9, 9
    print(f"Prob. size : nv={n_votes}, nc={n_cands}")

    V1 = np.array([np.random.permutation(n_cands) for _ in range(n_votes)])
    V2 = np.array([np.random.permutation(n_cands) for _ in range(n_votes)])
    # print(V1.argsort())
    # print(V2.argsort())

    # s = time.perf_counter()
    # res = dist.speard(V1, V2)
    # e = time.perf_counter()
    # t1 = e - s
    # print(f"Target(C++): res={res}, t={e-s:.4f}s")

    s = time.perf_counter()
    res = fastmap.bfcm.bfcm(V1.argsort().astype(np.int32), V2.argsort().astype(np.int32))
    e = time.perf_counter()
    t2 = e - s
    print(f"Custom C   : res={res}, t={e-s:.4f}s")

    s = time.perf_counter()
    D = np.abs(np.subtract.outer(V1.argsort(), V2.argsort())).swapaxes(1, 2)
    res = fastmap.proto.bfcm(V1, V2, D)
    e = time.perf_counter()
    t3 = e - s
    print(f"Numpy      : res={res}, t={e-s:.4f}s")
