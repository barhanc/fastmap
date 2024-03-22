import numpy as np
import time

import fastmap.bfcm
import fastmap.proto

import mapel.elections.distances.cppdistances as dist

if __name__ == "__main__":
    n_votes, n_cands = 9, 9
    print(f"Prob. size : nv={n_votes}, nc={n_cands}")

    V1 = np.array([np.random.permutation(n_cands) for _ in range(n_votes)])
    V2 = np.array([np.random.permutation(n_cands) for _ in range(n_votes)])
    # print(V1.argsort())
    # print(V2.argsort())

    s = time.perf_counter()
    res = dist.speard(V1, V2)
    e = time.perf_counter()
    t1 = e - s
    print(f"Target(C++): res={res}, t={e-s:.4f}s")

    s = time.perf_counter()
    # Easy numpy calculation of Spearman distance tensor. But data passing between numpy and C for
    # large tensor D is kinda slow. We should try passing V1 and V2 and calculating D in pure C
    # D = np.abs(np.subtract.outer(V1.argsort(), V2.argsort()), dtype=np.int32).swapaxes(1, 2)
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

    # s = time.perf_counter()
    # res = fastmap.proto.spear_ilp(V1, V2)
    # e = time.perf_counter()
    # t4 = e - s
    # print(f"ILP-1      : {res}, {e-s:.4f}s")

    s = time.perf_counter()
    res = int(fastmap.proto.spear_ilp2(V1, V2, solver := "CPLEX"))
    e = time.perf_counter()
    t5 = e - s
    print(f"ILP-2      : res={res}, t={e-s:.4f}s", f"[{solver}]")

    print(f"Speed increase (C)    : {t1/t2:.3f}x")
    print(f"Speed increase (Numpy): {t1/t3:.3f}x")
    # print(f"Speed increase (ILP-1): {t1/t4:.3f}x")
    print(f"Speed increase (ILP-2): {t1/t5:.3f}x")
