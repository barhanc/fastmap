import numpy as np
from itertools import product


def swap_fw(
    pos_U: np.ndarray[int, int],
    pos_V: np.ndarray[int, int],
    maxiter: int = 30,
    tol: float = 1e-4,
) -> int:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as e:
        raise ImportError("scipy.optimize module not found") from e

    def random_bistochastic(n: int, eps: float = 1e-8) -> np.ndarray[float, float]:
        X = np.random.random((n, n))
        while True:
            X /= X.sum(0)
            X = X / X.sum(1)[:, np.newaxis]
            rsum = X.sum(1)
            csum = X.sum(0)

            if np.all(np.abs(rsum - 1) < eps) and np.all(np.abs(csum - 1) < eps):
                break
        return X

    nv, nc = pos_U.shape
    D = np.array(
        [
            [
                np.multiply.outer(
                    np.subtract.outer(pos_U[i, :], pos_U[i, :]),
                    np.subtract.outer(pos_V[j, :], pos_V[j, :]),
                )
                < 0
                for j in range(nv)
            ]
            for i in range(nv)
        ]
    )
    D = D.swapaxes(3, 4)

    grad_f = np.zeros((nc, nc))
    P = random_bistochastic(nc)
    Q = np.zeros((nc, nc))

    for _ in range(maxiter):
        # Compute voters' assignment
        cost_nv = np.sum(D * np.multiply.outer(P, P), axis=(2, 3, 4, 5))
        rows_nv, cols_nv = linear_sum_assignment(cost_nv)

        # Compute gradient
        grad_f = np.sum(D[rows_nv, cols_nv, ...] * P, axis=(0, 3, 4))

        # Compute direction Q
        rows_nc, cols_nc = linear_sum_assignment(grad_f)
        Q[:, :] = 0
        Q[rows_nc, cols_nc] = 1

        # Compute the step size alpha
        R = Q - P
        S = np.multiply.outer(R, P)
        a = np.sum(D[rows_nv, cols_nv, ...] * np.multiply.outer(R, R))
        b = np.sum(D[rows_nv, cols_nv, ...] * (S + np.transpose(S, (2, 3, 0, 1))))

        if a > 0 and 0 <= -b / (2 * a) <= 1:
            alpha = -b / (2 * a)
        else:
            alpha = 0 if a + b > 0 else 1

        # Update P
        P_next = (1 - alpha) * P + alpha * Q
        norm2 = np.sum((P_next - P) ** 2)
        P = P_next

        if norm2 <= tol**2:
            break

    # Obtain solution
    rows_nc, cols_nc = linear_sum_assignment(P, maximize=True)
    P[:, :] = 0
    P[rows_nc, cols_nc] = 1

    cost_nv = np.sum(D * np.multiply.outer(P, P), axis=(2, 3, 4, 5))
    rows_nv, cols_nv = linear_sum_assignment(cost_nv)

    return int(0.5 * np.sum(cost_nv[rows_nv, cols_nv]))


ORDINAL_CULTURES = [
    {"id": "ic", "params": {}},
    {"id": "norm-mallows", "params": {"norm-phi": 0.05}},
    {"id": "norm-mallows", "params": {"norm-phi": 0.20}},
    {"id": "norm-mallows", "params": {"norm-phi": 0.50}},
    {"id": "urn", "params": {"alpha": 0.05}},
    {"id": "urn", "params": {"alpha": 0.20}},
    {"id": "urn", "params": {"alpha": 1.00}},
    {"id": "euclidean", "params": {"dim": 1, "space": "uniform"}},
    {"id": "euclidean", "params": {"dim": 2, "space": "uniform"}},
    {"id": "euclidean", "params": {"dim": 3, "space": "uniform"}},
    {"id": "euclidean", "params": {"dim": 10, "space": "uniform"}},
    {"id": "euclidean", "params": {"dim": 2, "space": "sphere"}},
    {"id": "euclidean", "params": {"dim": 3, "space": "sphere"}},
    {"id": "walsh", "params": {}},
    {"id": "conitzer", "params": {}},
    {"id": "spoc", "params": {}},
    {"id": "single-crossing", "params": {}},
    {"id": "group-separable", "params": {"tree_sampler": "caterpillar"}},
    {"id": "group-separable", "params": {"tree_sampler": "balanced"}},
]

ORDINAL_CULTURES = [
    {"id": "ic", "params": {}},
    {"id": "norm-mallows", "params": {"norm-phi": 0.05}},
    {"id": "norm-mallows", "params": {"norm-phi": 0.20}},
    {"id": "norm-mallows", "params": {"norm-phi": 0.50}},
    {"id": "urn", "params": {"alpha": 0.05}},
    {"id": "urn", "params": {"alpha": 0.20}},
    {"id": "urn", "params": {"alpha": 1.00}},
    {"id": "euclidean", "params": {"dim": 1, "space": "uniform"}},
    {"id": "euclidean", "params": {"dim": 2, "space": "uniform"}},
    {"id": "euclidean", "params": {"dim": 3, "space": "uniform"}},
    {"id": "euclidean", "params": {"dim": 10, "space": "uniform"}},
    {"id": "euclidean", "params": {"dim": 2, "space": "sphere"}},
    {"id": "euclidean", "params": {"dim": 3, "space": "sphere"}},
    {"id": "walsh", "params": {}},
    {"id": "conitzer", "params": {}},
    {"id": "spoc", "params": {}},
    {"id": "single-crossing", "params": {}},
    {"id": "group-separable", "params": {"tree_sampler": "caterpillar"}},
    {"id": "group-separable", "params": {"tree_sampler": "balanced"}},
]

import time
import random
import mapel.elections as mapel

nv, nc = 96, 8
culture1 = ORDINAL_CULTURES[-2]  # [random.randint(0, len(ORDINAL_CULTURES) - 1)]
culture2 = ORDINAL_CULTURES[-2]  # [random.randint(0, len(ORDINAL_CULTURES) - 1)]

print("ISOMORPHIC SWAP\n")
print(
    f"Candidates {nc} :: Votes {nv}\n"
    f"Culture1 {culture1['id']} {culture1['params']}\n"
    f"Culture2 {culture2['id']} {culture2['params']}\n"
)

U = mapel.generate_ordinal_election(
    culture_id=culture1["id"],
    num_candidates=nc,
    num_voters=nv,
    **culture1["params"],
)
V = mapel.generate_ordinal_election(
    culture_id=culture2["id"],
    num_candidates=nc,
    num_voters=nv,
    **culture2["params"],
)

t1 = time.monotonic()
d1, _ = mapel.compute_distance(U, V, distance_id="swap")
t1 = time.monotonic() - t1
print(f"Mapel :: {d1} :: Time {t1:6.3f}s")

t2 = time.monotonic()
pos_U, pos_V = U.votes.argsort(), V.votes.argsort()
d2 = swap_fw(pos_U, pos_V)
t2 = time.monotonic() - t2
print(f"FW   :: {d2} :: Time {t2:6.3f}s")
