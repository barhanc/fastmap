import sys
import csv
import time

import numpy as np
import fastmap
import mapel.elections as mapel

from itertools import product
from tqdm import tqdm

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

nv, nc = 50, 8

for culture1, culture2 in product(ORDINAL_CULTURES, ORDINAL_CULTURES):
    U = mapel.generate_ordinal_election(
        culture_id=culture1["id"], num_candidates=nc, num_voters=nv, **culture1["params"]
    )
    V = mapel.generate_ordinal_election(
        culture_id=culture2["id"], num_candidates=nc, num_voters=nv, **culture2["params"]
    )

    print(f"{culture1['id']}{culture1['params']}, {culture2['id']}{culture2['params']}")

    t1 = time.time()
    d1, _ = mapel.compute_distance(U, V, distance_id="swap")
    t1 = time.time() - t1
    print(f"Mapel :: {d1} :: Time {t1:6.3f}s")

    t2 = time.time()
    d2 = fastmap.swap(U.votes, V.votes, method="bf")
    t2 = time.time() - t2
    print(f"C(bf) :: {d2} :: Time {t2:6.3f}s :: Time ratio {t2 / t1:6.3f}")

    assert d1 == d2, "Wrong answer"

    t3 = time.time()
    d3 = min(fastmap.swap(U.votes, V.votes, method="aa") for _ in range(50))
    t3 = time.time() - t3
    print(f"C(aa) :: {d3} :: Time {t3:6.3f}s :: Time ratio {t3 / t1:6.3f} :: Approx. ratio {d3/d1:.3f}")
    print("=" * 60)
    print()
