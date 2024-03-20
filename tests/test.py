import time
import numpy as np

from fastmap.proto import bf_with_cand_match
import mapel.elections.distances.cppdistances as dist
from fastmap.fast import bfcm

if __name__ == "__main__":
    n_votes, n_cands = 8, 8
    V1 = np.array([np.random.permutation(n_cands) for _ in range(n_votes)])
    V2 = np.array([np.random.permutation(n_cands) for _ in range(n_votes)])

    s = time.perf_counter()
    res = dist.speard(V1, V2)
    e = time.perf_counter()
    print(f"Cpp TO BEAT (with no extensions): {res}, {e-s:.4f}")

    s = time.perf_counter()
    D = np.abs(np.subtract.outer(np.argsort(V1), np.argsort(V2))).swapaxes(1, 2).swapaxes(-2, -1)
    res = bfcm(D.flatten(), n_votes, n_cands)
    e = time.perf_counter()
    print(f"My CPP                          : {res}, {e-s:.4f}")

    s = time.perf_counter()
    pos1 = np.argsort(V1)
    pos2 = np.argsort(V2)
    D = np.abs(np.subtract.outer(pos1, pos2)).swapaxes(1, 2)
    res = bf_with_cand_match(V1, V2, D)
    e = time.perf_counter()
    print(f"BF with Cand. Match Precomputed D:{res}, {e-s:.4f}")
