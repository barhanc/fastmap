import time
import random
import fastmap
import mapel.elections as mapel
import numpy as np


APPROVAL_CULTURES = [
    {"id": "ic", "params": {"p": 0.10}},
    {"id": "ic", "params": {"p": 0.15}},
    {"id": "ic", "params": {"p": 0.50}},
    {"id": "euclidean", "params": {"dim": 1, "space": "uniform", "radius": 0.05}},
    {"id": "euclidean", "params": {"dim": 2, "space": "uniform", "radius": 0.20}},
    {"id": "euclidean", "params": {"dim": 1, "space": "gaussian", "radius": 0.05}},
    {"id": "euclidean", "params": {"dim": 2, "space": "gaussian", "radius": 0.20}},
    {"id": "resampling", "params": {"p": 0.10, "phi": 0.50}},
    {"id": "resampling", "params": {"p": 0.25, "phi": 0.75}},
]

nv, nc = 96, 8
culture1 = APPROVAL_CULTURES[random.randint(0, len(APPROVAL_CULTURES) - 1)]
culture2 = APPROVAL_CULTURES[random.randint(0, len(APPROVAL_CULTURES) - 1)]

U = mapel.generate_approval_election(culture_id=culture1["id"], num_candidates=nc, num_voters=nv, **culture1["params"])
V = mapel.generate_approval_election(culture_id=culture2["id"], num_candidates=nc, num_voters=nv, **culture2["params"])

oU, oV = np.zeros((nv, nc)), np.zeros((nv, nc))
for i, ballot in enumerate(U.votes):
    oU[i][list(ballot)] = 1
for i, ballot in enumerate(V.votes):
    oV[i][list(ballot)] = 1

print("\n\nISOMORPHIC HAMMING\n")
print(
    f"Candidates {nc} :: Votes {nv} :: Culture1 {culture1['id']} {culture1['params']} :: Culture2 {culture2['id']} {culture2['params']}\n"
)

t1 = time.monotonic()
d1, _ = mapel.compute_distance(U, V, distance_id="l1-approvalwise")
t1 = time.monotonic() - t1
print(f"Mapel :: {d1*nv:.2f} :: Time {t1:6.3f}s")

for trial in range(1):

    print(f"\nTrial: {trial}\n")

    t2 = time.monotonic()
    d2 = fastmap.hamming(oU, oV, method="bf")
    t2 = time.monotonic() - t2
    print(f"C(bf) :: {d2:.2f} :: Time {t2:6.3f}s :: Time ratio {t2 / t1:6.3f}")

    t3 = time.monotonic()
    d3 = fastmap.hamming(oU, oV, method="aa", repeats=300, seed=-1)
    t3 = time.monotonic() - t3
    print(f"C(aa) :: {d3:.2f} :: Time {t3:6.3f}s :: Time ratio {t3 / t1:6.3f} :: Approx ratio :: {d3 / d2:.3f}")

    t4 = time.monotonic()
    d4 = fastmap.hamming(oU, oV, method="bb", repeats=300, seed=-1)
    t4 = time.monotonic() - t4
    print(f"C(bb) :: {d4:.2f} :: Time {t4:6.3f}s :: Time ratio {t4 / t1:6.3f}")

    assert d2 == d4, "Wrong answer"
