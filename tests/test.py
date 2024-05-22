import time
import numpy as np

import mapel.elections as mapel
import fastmap


if __name__ == "__main__":
    print("ISOMORPHIC SPEARMAN\n")
    nv, nc = 10, 10
    culture_id = "ic"
    print(f"Candidates {nc} :: Votes {nv} :: Culture {culture_id}\n")

    U = mapel.generate_ordinal_election(culture_id=culture_id, num_candidates=nc, num_voters=nv)
    V = mapel.generate_ordinal_election(culture_id=culture_id, num_candidates=nc, num_voters=nv)

    t1 = time.time()
    d1, _ = mapel.compute_distance(U, V, distance_id="spearman")
    t1 = time.time() - t1
    print(f"Mapel :: {d1} :: Time {t1:6.3f}s")

    t2 = time.time()
    d2 = fastmap.spearman(U.votes, V.votes, method="bf")
    t2 = time.time() - t2
    print(f"C(bf) :: {d2} :: Time {t2:6.3f}s :: Time ratio {t2 / t1:6.3f}")

    assert d1 == d2, "Wrong answer"

    t3 = time.time()
    restarts = 50
    d3 = min(results := [fastmap.spearman(U.votes, V.votes, method="aa") for _ in range(restarts)])
    t3 = time.time() - t3
    print(f"C(aa) :: {d3} :: Time {t3:6.3f}s :: Time ratio {t3 / t1:6.3f} :: Approx ratio :: {d3 / d1:.3f}")

    # ==============================================================================================
    # ==============================================================================================

    print("\n\nISOMORPHIC HAMMING\n")
    nv, nc = 50, 7
    culture_id = "ic"
    print(f"Candidates {nc} :: Votes {nv} :: Culture {culture_id}\n")

    U = mapel.generate_approval_election(culture_id=culture_id, num_candidates=nc, num_voters=nv)
    V = mapel.generate_approval_election(culture_id=culture_id, num_candidates=nc, num_voters=nv)

    t1 = time.time()
    d1, _ = mapel.compute_distance(U, V, distance_id="l1-approvalwise")
    t1 = time.time() - t1
    print(f"Mapel :: {d1*nv:.2f} :: Time {t1:6.3f}s")

    oU, oV = np.zeros((nv, nc)), np.zeros((nv, nc))
    for i, ballot in enumerate(U.votes):
        oU[i][list(ballot)] = 1
    for i, ballot in enumerate(V.votes):
        oV[i][list(ballot)] = 1

    t2 = time.time()
    d2 = fastmap.hamming(oU, oV, method="bf")
    t2 = time.time() - t2
    print(f"C(bf) :: {d2:.2f} :: Time {t2:6.3f}s :: Time ratio {t2 / t1:6.3f}")

    t3 = time.time()
    restarts = 50
    d3 = min(results := [fastmap.hamming(oU, oV, method="aa") for _ in range(restarts)])
    t3 = time.time() - t3
    print(
        f"C(aa) :: {d3:.2f} :: Time {t3:6.3f}s :: Time ratio {t3 / t1:6.3f} :: Approx ratio :: {d3 / d2 if d2 != 0 else 1.0:.3f}"
    )
