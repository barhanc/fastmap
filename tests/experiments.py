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

FASTMAP_DIST = {"hamming": fastmap.hamming, "swap": fastmap.swap, "spearman": fastmap.spearman}


def experiment(dist_id: str, nc: int = 8, nvs: list[int] = [10, 20, 50, 100, 200], samples: int = 30):
    if dist_id in {"hamming"}:
        cultures = APPROVAL_CULTURES
        generate_election = lambda **kwargs: mapel.generate_approval_election(**kwargs)
    elif dist_id in {"swap", "spearman"}:
        cultures = ORDINAL_CULTURES
        generate_election = lambda **kwargs: mapel.generate_ordinal_election(**kwargs)
    else:
        raise ValueError(f"Unknown distance id: {dist_id}")

    def transform(U, nv, nc, dist_id):
        if dist_id not in {"hamming"}:
            return U.votes
        oU = np.zeros((nv, nc))
        for i, ballot in enumerate(U.votes):
            oU[i][list(ballot)] = 1
        return oU

    results = []
    with open("results.csv", "w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(
            [
                "nc",
                "nv",
                "culture_id",
                "culture_params",
                "sample",
                "Mapel time[s]",
                "C_bf time[s]",
                "C_aa time[s]",
                "Mapel dist",
                "C_bf dist",
                "C_aa dist",
            ]
        )

    tot = len(nvs) * len(cultures) * samples
    for nv, culture, sample in (pbar := tqdm(product(nvs, cultures, range(samples)), total=tot)):
        # Update progress bar
        pbar.set_description(f"{nv},{culture['id']}, {sample}")

        U = generate_election(culture_id=culture["id"], num_candidates=nc, num_voters=nv, **culture["params"])
        V = generate_election(culture_id=culture["id"], num_candidates=nc, num_voters=nv, **culture["params"])

        # Mapel Spearman
        t1 = time.perf_counter()
        d1, _ = mapel.compute_distance(U, V, distance_id=dist_id if dist_id != "hamming" else "l1-approvalwise")
        t1 = time.perf_counter() - t1

        fU, fV = transform(U, nv, nc, dist_id), transform(V, nv, nc, dist_id)
        # C Spearman
        t2 = time.perf_counter()
        d2 = FASTMAP_DIST[dist_id](fU, fV, method="bf")
        t2 = time.perf_counter() - t2

        if dist_id != "hamming":
            assert d1 == d2, "Wrong answer"
        if dist_id == "hamming":
            d1 = int(d1 * nv)

        t3 = time.perf_counter()
        d3 = min(FASTMAP_DIST[dist_id](fU, fV, method="aa") for _ in range(50))
        t3 = time.perf_counter() - t3

        results.append([nc, nv, culture["id"], culture["params"], sample, t1, t2, t3, d1, d2, d3])

    with open("results.csv", "a", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        for row in results:
            writer.writerow(row)


if __name__ == "__main__":
    argv = sys.argv
    if len(argv) != 2:
        print("Usage: python3 experiments.py [spearman|swap|hamming]")
        exit(0)
    dist_id = argv[1]
    experiment(dist_id=dist_id)
