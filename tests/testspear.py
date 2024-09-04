import time
import fastmap
import mapel.elections as mapel
import numpy as np

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

nv, nc = 10, 10
culture1 = ORDINAL_CULTURES[0]  # random.randint(0, len(ORDINAL_CULTURES) - 1)]
culture2 = ORDINAL_CULTURES[0]  # random.randint(0, len(ORDINAL_CULTURES) - 1)]

U = mapel.generate_ordinal_election(culture_id=culture1["id"], num_candidates=nc, num_voters=nv, **culture1["params"])
V = mapel.generate_ordinal_election(culture_id=culture2["id"], num_candidates=nc, num_voters=nv, **culture2["params"])

print("ISOMORPHIC SPEARMAN\n")
print(
    f"Candidates {nc} :: Votes {nv} :: Culture1 {culture1['id']} {culture1['params']} :: Culture2 {culture2['id']} {culture2['params']}\n"
)

t1 = time.monotonic()
d1, _ = mapel.compute_distance(U, V, distance_id="spearman")
t1 = time.monotonic() - t1
print(f"Mapel :: {d1} :: Time {t1:6.3f}s")

for trial in range(5):
    print(f"\nTrial: {trial}\n")

    t2 = time.monotonic()
    d2 = fastmap.spearman(U.votes, V.votes, method="bf")
    t2 = time.monotonic() - t2
    print(f"C(bf) :: {d2} :: Time {t2:6.3f}s :: Time ratio {t2 / t1:6.3f}")

    assert d1 == d2, "Wrong answer"

    t3 = time.monotonic()
    d3 = fastmap.spearman(U.votes, V.votes, method="aa", repeats=300, seed=42)
    t3 = time.monotonic() - t3
    print(f"C(aa) :: {d3} :: Time {t3:6.3f}s :: Time ratio {t3 / t1:6.3f} :: Approx ratio :: {d3 / d1:.3f}")

    t4 = time.monotonic()
    d4 = fastmap.spearman(U.votes, V.votes, method="bb", repeats=300, seed=42)
    t4 = time.monotonic() - t4
    print(f"C(bb) :: {d4} :: Time {t4:6.3f}s :: Time ratio {t4 / t1:6.3f}")

    assert d1 == d4, "Wrong answer"
