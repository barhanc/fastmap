import time
import numpy as np

import fastmap.example
import fastmap.example.wrapper as mdist
import fastmap.isomorphic

from mapel.elections.distances import cppdistances as dist

from itertools import permutations


def hamm_bf(U: np.ndarray[bool | int], V: np.ndarray[bool | int]) -> int:
    assert U.shape == V.shape
    assert len(U.shape) == 2

    nv, nc = U.shape
    candidates = np.arange(nc)
    U, V = U.astype(bool), V.astype(bool)

    best_res = float("inf")
    for nu in permutations(range(nv)):
        for sigma in permutations(range(nc)):
            sigma = np.array(sigma)
            res = 0
            for i in range(nv):
                u = set(sigma[candidates[U[i]]])
                v = set(candidates[V[nu[i]]])
                res += len(u.symmetric_difference(v))
            best_res = min(best_res, res)

    return best_res


if __name__ == "__main__":
    nv, nc = 100, 7
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

    nv, nc = 4, 3
    print(f"Prob. size : nv={nv}, nc={nc}")
    print("Hamming")

    U = np.array([np.random.randint(0, 2, size=nc) for _ in range(nv)])
    V = np.array([np.random.randint(0, 2, size=nc) for _ in range(nv)])

    print(U)
    print(V)

    res = mdist.hamming(U, V)
    print(res)

    res = hamm_bf(U, V)
    print(res)
