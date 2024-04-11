import time
import numpy as np

import fastmap.example.wrapper as mdist
from mapel.elections.distances import cppdistances as dist


def hamm_bf(U: np.ndarray[bool | int], V: np.ndarray[bool | int]) -> int:
    from itertools import permutations

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
    print("=" * 60 + "\nSpearman\n" + "=" * 60)
    nv, nc = 10, 10
    print(f"Prob. size : nv={nv}, nc={nc}\n")
    U = np.array([np.random.permutation(nc) for _ in range(nv)])
    V = np.array([np.random.permutation(nc) for _ in range(nv)])

    t1 = time.perf_counter()
    res1 = dist.speard(U, V)
    t1 = time.perf_counter() - t1
    print(f"Mapel           : res={res1:4d}, t={t1:.4f}s")

    t2 = time.perf_counter()
    res2 = mdist.spearman(U, V, method="bf")
    t2 = time.perf_counter() - t2
    print(f"C - bf          : res={res2:4d}, t={t2:.4f}s")

    assert res2 == res1

    t3 = time.perf_counter()
    mss = 10
    res3 = min(mdist.spearman(U, V, method="aa") for _ in range(mss))
    t3 = time.perf_counter() - t3
    print(f"C - aa mss={mss:5d}: res={res3:4d}, t={t3:.4f}s")

    print(f"\nSpeed up Mapel vs C-bf: {t1 / t2:4.2f}x")
    print(f"\nSpeed up C-bf  vs C-aa: {t2 / t3:4.2f}x")
    print(f"\nApprox ratio          : {res3 / res2:.2f}")

    # ==============================================================================================

    print("\n\n" + "=" * 60 + "\nHamming\n" + "=" * 60)
    nv, nc = 6, 7
    print(f"Prob. size : nv={nv}, nc={nc}")
    U = np.array([np.random.randint(0, 2, size=nc) for _ in range(nv)])
    V = np.array([np.random.randint(0, 2, size=nc) for _ in range(nv)])

    t1 = time.perf_counter()
    res1 = hamm_bf(U, V)
    t1 = time.perf_counter() - t1
    print(f"Python     : res={res1:4d}, t={t1:.4f}s")

    t2 = time.perf_counter()
    res2 = mdist.hamming(U, V, method="bf")
    t2 = time.perf_counter() - t2
    print(f"C - bf     : res={res2:4d}, t={t2:.4f}s")

    assert res2 == res1

    t3 = time.perf_counter()
    mss = 50
    res3 = min(mdist.hamming(U, V, method="aa") for _ in range(mss))
    t3 = time.perf_counter() - t3
    print(f"C - aa mss={mss:5d}: res={res3:4d}, t={t3:.4f}s")
