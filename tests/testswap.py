import time
import numpy as np

import mapel.elections as mapel
import fastmap
import warnings

if __name__ == "__main__":
    print("ISOMORPHIC SWAP\n")
    nv, nc = 100, 8
    culture_id = "urn"
    print(f"Candidates {nc} :: Votes {nv} :: Culture {culture_id}\n")

    U = mapel.generate_ordinal_election(culture_id=culture_id, num_candidates=nc, num_voters=nv)
    V = mapel.generate_ordinal_election(culture_id=culture_id, num_candidates=nc, num_voters=nv)

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
