import time
import random
import numpy as np

import mapel.elections as mapel
import fastmap


if __name__ == "__main__":

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
    culture2 = ORDINAL_CULTURES[4]  # random.randint(0, len(ORDINAL_CULTURES) - 1)]

    print("ISOMORPHIC SPEARMAN\n")
    print(f"Candidates {nc} :: Votes {nv} :: Culture1 {culture1['id']} :: Culture2 {culture2['id']}\n")

    U = mapel.generate_ordinal_election(
        culture_id=culture1["id"], num_candidates=nc, num_voters=nv, **culture1["params"]
    )
    V = mapel.generate_ordinal_election(
        culture_id=culture2["id"], num_candidates=nc, num_voters=nv, **culture2["params"]
    )

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

    t4 = time.time()
    d4 = fastmap.spearman(U.votes, V.votes, method="bb")
    t4 = time.time() - t4
    print(f"C(bb) :: {d4} :: Time {t4:6.3f}s :: Time ratio {t4 / t1:6.3f}")

    assert d1 == d4, "Wrong answer"

    # ==============================================================================================
    # ==============================================================================================

    APPROVAL_CULTURES = [
        {"id": "ic", "params": {"p": 0.1}},
        {"id": "ic", "params": {"p": 0.2}},
        {"id": "euclidean", "params": {"dim": 1, "space": "uniform", "radius": 0.05}},
        {"id": "euclidean", "params": {"dim": 2, "space": "uniform", "radius": 0.20}},
        {"id": "euclidean", "params": {"dim": 1, "space": "gaussian", "radius": 0.05}},
        {"id": "euclidean", "params": {"dim": 2, "space": "gaussian", "radius": 0.20}},
        {"id": "resampling", "params": {"p": 0.10, "phi": 0.50}},
        {"id": "resampling", "params": {"p": 0.25, "phi": 0.75}},
    ]
    nv, nc = 11, 11
    culture1 = APPROVAL_CULTURES[random.randint(0, len(APPROVAL_CULTURES) - 1)]
    culture2 = APPROVAL_CULTURES[random.randint(0, len(APPROVAL_CULTURES) - 1)]

    print("\n\nISOMORPHIC HAMMING\n")
    print(f"Candidates {nc} :: Votes {nv} :: Culture1 {culture1['id']} :: Culture2 {culture2['id']}\n")

    U = mapel.generate_approval_election(
        culture_id=culture1["id"], num_candidates=nc, num_voters=nv, **culture1["params"]
    )
    V = mapel.generate_approval_election(
        culture_id=culture2["id"], num_candidates=nc, num_voters=nv, **culture2["params"]
    )

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

    t4 = time.time()
    d4 = fastmap.hamming(oU, oV, method="bb")
    t4 = time.time() - t4
    print(f"C(bb) :: {d4:.2f} :: Time {t4:6.3f}s :: Time ratio {t4 / t1:6.3f}")

    assert d2 == d4, "Wrong answer"
