import time
import random
import fastmap
import mapel.elections as mapel
import numpy as np

ORDINAL_CULTURES = [
    {"id": "ic", "params": {}},
    {"id": "norm-mallows", "params": {"norm-phi": 0.05}},
    {"id": "norm-mallows", "params": {"norm-phi": 0.20}},
    {"id": "norm-mallows", "params": {"norm-phi": 0.50}},
    {"id": "urn", "params": {"alpha": 0.05}},
    {"id": "urn", "params": {"alpha": 0.20}},
    {"id": "urn", "params": {"alpha": 1.00}},
    {"id": "euclidean", "params": {"dim": 1, "space": "uniform"}},
    {"id": "euclidean", "params": {"dim": 2, "space": "uniform"}},
    {"id": "euclidean", "params": {"dim": 3, "space": "uniform"}},
    {"id": "euclidean", "params": {"dim": 10, "space": "uniform"}},
    {"id": "euclidean", "params": {"dim": 2, "space": "sphere"}},
    {"id": "euclidean", "params": {"dim": 3, "space": "sphere"}},
    {"id": "walsh", "params": {}},
    {"id": "conitzer", "params": {}},
    {"id": "spoc", "params": {}},
    {"id": "single-crossing", "params": {}},
    {"id": "group-separable", "params": {"tree_sampler": "caterpillar"}},
    {"id": "group-separable", "params": {"tree_sampler": "balanced"}},
]


def dist(M_U, M_V, sigma):
    M_V = M_V[sigma, :]
    M_V = M_V[:, sigma]
    return np.sum(np.abs(M_U - M_V))


def bf(M_U: np.ndarray, M_V: np.ndarray):
    from itertools import permutations

    assert M_U.shape == M_V.shape
    nc = M_U.shape[0]

    best = float("inf")
    for sigma in permutations(range(nc)):
        best = min(best, dist(M_U, M_V, sigma))

    return best


nv, nc = 100, 8
culture1 = ORDINAL_CULTURES[0]  # random.randint(0, len(ORDINAL_CULTURES) - 1)]
culture2 = ORDINAL_CULTURES[0]  # random.randint(0, len(ORDINAL_CULTURES) - 1)]

print(
    "\nPAIRWISE\n\n"
    f"Candidates {nc} :: Votes {nv} :: "
    f"Culture1 {culture1['id']} {culture1['params']} :: "
    f"Culture2 {culture2['id']} {culture2['params']}\n"
)

U = mapel.generate_ordinal_election(
    culture_id=culture1["id"],
    num_candidates=nc,
    num_voters=nv,
    **culture1["params"],
)
V = mapel.generate_ordinal_election(
    culture_id=culture2["id"],
    num_candidates=nc,
    num_voters=nv,
    **culture2["params"],
)


t1 = time.monotonic()
d1 = bf(U.votes_to_pairwise_matrix(), V.votes_to_pairwise_matrix())
t1 = time.monotonic() - t1
print(f"Python :: {d1:6.3f} :: Time {t1:6.3f}s")

for trial in range(1):

    print(f"\nTrial : {trial}\n")

    t2 = time.monotonic()
    d2 = fastmap.pairwise(
        U.votes_to_pairwise_matrix(),
        V.votes_to_pairwise_matrix(),
        method="faq",
        repeats=300,
        maxiter=30,
        tol=1e-4,
    )
    t2 = time.monotonic() - t2
    print(f"C(faq) :: {d2:6.3f} :: Time {t2:6.3f}s :: Approx. ratio {d2 / d1 if d1 > 0 else d1 == d2:.3f}")

    assert round(d1, 6) <= round(d2, 6), f"Wrong answer: {d1} > {d2}"

    t3 = time.monotonic()
    d3 = fastmap.pairwise(
        U.votes_to_pairwise_matrix(),
        V.votes_to_pairwise_matrix(),
        method="aa",
        repeats=300,
    )
    t3 = time.monotonic() - t3
    print(f"C(aa)  :: {d3:6.3f} :: Time {t3:6.3f}s :: Approx. ratio {d3 / d1 if d1 > 0 else d1 == d3:.3f}")

    assert round(d1, 6) <= round(d3, 6), f"Wrong answer: {d1} > {d3}"
