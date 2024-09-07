import random
import concurrent.futures

import fastmap
import numpy as np
import matplotlib.pyplot as plt
from mapel.elections import generate_ordinal_election, OrdinalElection

from sklearn import manifold
from tqdm import tqdm

seed = 42
cultures = [
    {
        "id": "ic",
        "params": {},
        "plot": {
            "name": "IC",
            "marker": "p",
            "color": "black",
            "alpha": 0.8,
        },
    },
    {
        "id": "norm-mallows",
        "params": {
            "norm-phi": 0.05,
        },
        "plot": {
            "name": "Mallows norm-φ=0.05",
            "marker": "p",
            "color": "brown",
            "alpha": 0.2,
        },
    },
    {
        "id": "norm-mallows",
        "params": {
            "norm-phi": 0.20,
        },
        "plot": {
            "name": "Mallows norm-φ=0.20",
            "marker": "p",
            "color": "brown",
            "alpha": 0.5,
        },
    },
    {
        "id": "norm-mallows",
        "params": {
            "norm-phi": 0.50,
        },
        "plot": {
            "name": "Mallows norm-φ=0.50",
            "marker": "p",
            "color": "brown",
            "alpha": 0.8,
        },
    },
    {
        "id": "urn",
        "params": {
            "alpha": 0.05,
        },
        "plot": {
            "name": "Urn α=0.05",
            "marker": "p",
            "color": "orange",
            "alpha": 0.2,
        },
    },
    {
        "id": "urn",
        "params": {
            "alpha": 0.20,
        },
        "plot": {
            "name": "Urn α=0.20",
            "marker": "p",
            "color": "orange",
            "alpha": 0.5,
        },
    },
    {
        "id": "urn",
        "params": {
            "alpha": 1.00,
        },
        "plot": {
            "name": "Urn α=1.00",
            "marker": "p",
            "color": "orange",
            "alpha": 0.8,
        },
    },
    {
        "id": "euclidean",
        "params": {
            "dim": 1,
            "space": "uniform",
        },
        "plot": {
            "name": "Interval",
            "marker": "s",
            "color": "deepskyblue",
            "alpha": 0.8,
        },
    },
    {
        "id": "euclidean",
        "params": {
            "dim": 2,
            "space": "uniform",
        },
        "plot": {
            "name": "Square",
            "marker": "s",
            "color": "cornflowerblue",
            "alpha": 0.8,
        },
    },
    {
        "id": "euclidean",
        "params": {
            "dim": 3,
            "space": "uniform",
        },
        "plot": {
            "name": "Cube",
            "marker": "s",
            "color": "royalblue",
            "alpha": 0.8,
        },
    },
    {
        "id": "euclidean",
        "params": {
            "dim": 10,
            "space": "uniform",
        },
        "plot": {
            "name": "10D-Cube",
            "marker": "s",
            "color": "darkblue",
            "alpha": 0.8,
        },
    },
    {
        "id": "euclidean",
        "params": {
            "dim": 2,
            "space": "sphere",
        },
        "plot": {
            "name": "Circle",
            "marker": "o",
            "color": "deeppink",
            "alpha": 0.8,
        },
    },
    {
        "id": "euclidean",
        "params": {
            "dim": 3,
            "space": "sphere",
        },
        "plot": {
            "name": "Sphere",
            "marker": "o",
            "color": "purple",
            "alpha": 0.8,
        },
    },
    {
        "id": "walsh",
        "params": {},
        "plot": {
            "name": "Walsh",
            "marker": "^",
            "color": "forestgreen",
            "alpha": 0.8,
        },
    },
    {
        "id": "conitzer",
        "params": {},
        "plot": {
            "name": "Conitzer",
            "marker": "^",
            "color": "limegreen",
            "alpha": 0.8,
        },
    },
    {
        "id": "spoc",
        "params": {},
        "plot": {
            "name": "SPOC",
            "marker": "^",
            "color": "cyan",
            "alpha": 0.8,
        },
    },
    {
        "id": "single-crossing",
        "params": {},
        "plot": {
            "name": "Single Crossing",
            "marker": "^",
            "color": "darkseagreen",
            "alpha": 0.8,
        },
    },
    {
        "id": "group-separable",
        "params": {
            "tree_sampler": "caterpillar",
        },
        "plot": {
            "name": "GS Caterpillar",
            "marker": "^",
            "color": "olivedrab",
            "alpha": 0.8,
        },
    },
    {
        "id": "group-separable",
        "params": {
            "tree_sampler": "balanced",
        },
        "plot": {
            "name": "GS Balanced",
            "marker": "^",
            "color": "olive",
            "alpha": 0.8,
        },
    },
]


def generate(cultures: list[dict], nv: int, nc: int, size: int, seed: int = 0):
    return (
        [
            (
                i * size + r,
                generate_ordinal_election(
                    culture_id=cultures[i]["id"],
                    num_candidates=nc,
                    num_voters=nv,
                    **cultures[i]["params"],
                    seed=seed + r,
                ),
            )
            for i in range(len(cultures))
            for r in range(size)
        ],
        {i * size + r: f"{cultures[i]['plot']['name']}" for i in range(len(cultures)) for r in range(size)},
    )


def f(t: tuple[int, OrdinalElection, int, OrdinalElection]):
    i, U, j, V = t
    return i, j, fastmap.swap(U.votes, V.votes, method="aa", repeats=30, seed=42)


def main():
    nv, nc = 100, 20
    size = 16
    args, desc = generate(cultures, nv, nc, size, seed)
    args = [(*args[i], *args[j]) for i in range(len(args)) for j in range(i + 1, len(args))]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(f, args), total=len(args)))

    # Create 2d embedding
    # -----------------------------------------------------

    X = np.zeros((len(cultures) * size, len(cultures) * size))

    for i, j, d in results:
        X[i, j] = d
        X[j, i] = d

    npos = manifold.MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=0,
        n_jobs=-1,
        normalized_stress="auto",
    ).fit_transform(X)

    # Plot map
    # -----------------------------------------------------

    for i in range(len(cultures)):
        plt.scatter(
            npos[i * size : i * size + size, 0],
            npos[i * size : i * size + size, 1],
            marker=cultures[i]["plot"]["marker"],
            color=cultures[i]["plot"]["color"],
            alpha=cultures[i]["plot"]["alpha"],
            edgecolors=cultures[i]["plot"]["color"],
            label=desc[i * size],
        )
    plt.title(f"Map of elections, nc={nc}, nv={nv}, Swap AA")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.axis("off")
    plt.savefig(f"./tests/map{random.randint(1, 10000)}.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
