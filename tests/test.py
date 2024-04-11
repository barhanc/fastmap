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
    print(f"Prob. size : nv={nv}, nc={nc}")
    U = np.array([np.random.permutation(nc) for _ in range(nv)])
    V = np.array([np.random.permutation(nc) for _ in range(nv)])

    s = time.perf_counter()
    stdres = dist.speard(U, V)
    e = time.perf_counter()
    t1 = e - s
    print(f"Mapel  : res={stdres:4d}, t={e-s:.4f}s")

    s = time.perf_counter()
    res1 = mdist.spearman(U, V, method="bf")
    e = time.perf_counter()
    t1 = e - s
    print(f"C - bf : res={res1:4d}, t={e-s:.4f}s")

    s = time.perf_counter()
    res2 = mdist.spearman(U, V, method="aa")
    e = time.perf_counter()
    t2 = e - s
    print(f"C - aa : res={res2:4d}, t={e-s:.4f}s")

    # print(f"Speed up: {t1 / t2:4.2f}x")
    # print(f"Approx ratio: {res2 / res1:.2f}")

    # ==============================================================================================

    # print("\n\n" + "=" * 60 + "\nHamming\n" + "=" * 60)
    # nv, nc = 1000, 1000
    # print(f"Prob. size : nv={nv}, nc={nc}")
    # U = np.array([np.random.randint(0, 2, size=nc) for _ in range(nv)])
    # V = np.array([np.random.randint(0, 2, size=nc) for _ in range(nv)])

    # s = time.time()
    # res = mdist.hamming(U, V, method="aa")
    # e = time.time()
    # print(f"C - bf     : res={res:4d}, t={e-s:.4f}s")

    # # res = hamm_bf(U, V)
    # # print(res)

    # print()
    # t2 = 0
    # best = float("inf")
    # for _ in range(10):
    #     s = time.perf_counter()
    #     res = mdist.hamming(U, V, method="aa")
    #     e = time.perf_counter()
    #     best = min(best, res)
    #     t2 += e - s
    #     print(f"C - aa     : res={res:4d}, t={e-s:.4f}s")
    # print("-" * 40)
    # print(f"C - aa Best: res={best:4d}")
    # print()
