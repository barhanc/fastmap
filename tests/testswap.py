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


if __name__ == "__main__":
    nv, nc = 10, 10
    random.seed(42)
    culture1 = ORDINAL_CULTURES[random.randint(0, len(ORDINAL_CULTURES) - 1)]
    culture2 = ORDINAL_CULTURES[random.randint(0, len(ORDINAL_CULTURES) - 1)]

    print("ISOMORPHIC SWAP\n")
    print(
        f"Candidates {nc} :: Votes {nv} :: Culture1 {culture1['id']} {culture1['params']} :: Culture2 {culture2['id']} {culture2['params']}\n"
    )

    U = mapel.generate_ordinal_election(
        culture_id=culture1["id"], num_candidates=nc, num_voters=nv, **culture1["params"]
    )
    V = mapel.generate_ordinal_election(
        culture_id=culture2["id"], num_candidates=nc, num_voters=nv, **culture2["params"]
    )

    t1 = time.time()
    d1, _ = mapel.compute_distance(U, V, distance_id="swap")
    t1 = time.time() - t1
    print(f"Mapel :: {d1} :: Time {t1:6.3f}s")

    t2 = time.time()
    d2 = fastmap.swap(U.votes, V.votes, method="bf")
    t2 = time.time() - t2
    print(f"C(bf) :: {d2} :: Time {t2:6.3f}s :: Time ratio {t2 / t1:6.3f}")

    assert d1 == d2, "Wrong answer"

    # t3 = time.time()
    # d3 = fastmap.swap(U.votes, V.votes, method="aa")
    # t3 = time.time() - t3
    # print(f"C(aa) :: {d3} :: Time {t3:6.3f}s :: Time ratio {t3 / t1:6.3f} :: Approx. ratio {d3/d1:.3f}")
