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

nv, nc = 50, 6
culture1 = ORDINAL_CULTURES[0]  # random.randint(0, len(ORDINAL_CULTURES) - 1)]
culture2 = ORDINAL_CULTURES[0]  # random.randint(0, len(ORDINAL_CULTURES) - 1)]

print("ISOMORPHIC SPEARMAN\n")
print(
    f"Candidates {nc} :: Votes {nv}\n"
    f"Culture1 {culture1['id']} {culture1['params']}\n"
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
d1, _ = mapel.compute_distance(U, V, distance_id="spearman")
t1 = time.monotonic() - t1
print(f"Mapel :: {d1} :: Time {t1:6.3f}s")

for trial in range(1):
    print(f"\nTrial: {trial}\n")

    t2 = time.monotonic()
    d2 = fastmap.spearman(U.votes, V.votes, method="bf")
    t2 = time.monotonic() - t2
    print(f"C(bf) :: {d2} :: Time {t2:6.3f}s :: Time ratio {t2 / t1:6.3f}")

    assert d1 == d2, "Wrong answer"

    t3 = time.monotonic()
    d3 = fastmap.spearman(U.votes, V.votes, method="aa", repeats=300, seed=-1)
    t3 = time.monotonic() - t3
    print(f"C(aa) :: {d3} :: Time {t3:6.3f}s :: Time ratio {t3 / t1:6.3f} :: Approx ratio :: {d3 / d1:.3f}")

    assert d1 <= d3, "Wrong answer"

    t4 = time.monotonic()
    d4 = fastmap.spearman(U.votes, V.votes, method="bb", repeats=300, seed=-1)
    t4 = time.monotonic() - t4
    print(f"C(bb) :: {d4} :: Time {t4:6.3f}s :: Time ratio {t4 / t1:6.3f}")

    assert d1 == d4, "Wrong answer"
