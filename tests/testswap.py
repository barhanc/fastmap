import time
import numpy as np

import mapel.elections as mapel
import fastmap.example.wrapper as cdist


nv, nc = 80, 8
culture_id = "ic"
print(f"\nCandidates {nc} :: Votes {nv} :: Culture {culture_id}\n")

U = mapel.generate_ordinal_election(culture_id=culture_id, num_candidates=nc, num_voters=nv)
V = mapel.generate_ordinal_election(culture_id=culture_id, num_candidates=nc, num_voters=nv)

t1 = time.time()
d1, _ = mapel.compute_distance(U, V, distance_id="swap")
t1 = time.time() - t1
print(f"Mapel :: {d1} :: Time {t1:6.3f}s")

t2 = time.time()
d2 = cdist.swap(U.votes, V.votes, method="bf")
t2 = time.time() - t2
print(f"C(bf) :: {d2} :: Time {t2:6.3f}s :: Time ratio {t2 / t1:6.3f}")

assert d1 == d2, "Wrong answer"
