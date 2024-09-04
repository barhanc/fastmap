import time, random
import fastmap
import mapel.elections as mapel

ORDINAL_CULTURES = [
    {"id": "ic", "params": {}},
    {"id": "mallows", "params": {"phi": 0.2}},
    {"id": "mallows", "params": {"phi": 0.5}},
    {"id": "mallows", "params": {"phi": 0.8}},
    {"id": "urn", "params": {"alpha": 0.1}},
    {"id": "urn", "params": {"alpha": 0.2}},
    {"id": "urn", "params": {"alpha": 0.5}},
    {"id": "euclidean", "params": {"dim": 1, "space": "uniform"}},
    {"id": "euclidean", "params": {"dim": 2, "space": "uniform"}},
    {"id": "conitzer", "params": {}},
    {"id": "walsh", "params": {}},
]

import numpy as np
from itertools import permutations


def bf(M_U: np.ndarray, M_V: np.ndarray):
    assert M_U.shape == M_V.shape
    nc, _ = M_U.shape
    best = float("inf")

    for sigma in permutations(range(nc)):
        res = 0
        for i in range(nc):
            for j in range(nc):
                res += abs(M_U[i, j] - M_V[sigma[i], sigma[j]])
        best = min(best, res)

    return best


nv, nc = 100, 9
culture1 = ORDINAL_CULTURES[random.randint(0, len(ORDINAL_CULTURES) - 1)]
culture2 = ORDINAL_CULTURES[random.randint(0, len(ORDINAL_CULTURES) - 1)]

U = mapel.generate_ordinal_election(
    culture_id=culture1["id"],
    num_candidates=nc,
    num_voters=nv,
    **culture1["params"],
    seed=0,
)
V = mapel.generate_ordinal_election(
    culture_id=culture2["id"],
    num_candidates=nc,
    num_voters=nv,
    **culture2["params"],
    seed=0,
)

print("PAIRWISE\n")
print(f"Candidates {nc} :: Votes {nv} :: Culture1 {culture1['id']} :: Culture2 {culture2['id']}\n")

t1 = time.monotonic()
d1 = bf(U.votes_to_pairwise_matrix(), V.votes_to_pairwise_matrix())
t1 = time.monotonic() - t1
print(f"Python :: {d1:6.3f} :: Time {t1:6.3f}")

for trial in range(5):

    print(f"\nTrial : {trial}\n")

    t2 = time.monotonic()
    d2 = fastmap.pairwise(
        U.votes_to_pairwise_matrix(), V.votes_to_pairwise_matrix(), method="faq", repeats=300, tol=1e-8, seed=0
    )
    t2 = time.monotonic() - t2
    print(f"C(faq) :: {d2:6.3f} :: Time {t2:6.3f}s :: Approx. ratio {d2 / d1 if d1 > 0 else d1 == d2:.3f}")

    t3 = time.monotonic()
    d3 = fastmap.pairwise(U.votes_to_pairwise_matrix(), V.votes_to_pairwise_matrix(), method="aa", repeats=300, seed=0)
    t3 = time.monotonic() - t3
    print(f"C(aa)  :: {d3:6.3f} :: Time {t3:6.3f}s :: Approx. ratio {d3 / d1 if d1 > 0 else d1 == d3:.3f}")
