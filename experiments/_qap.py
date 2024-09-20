import numpy as np


def qap_faq(D: np.ndarray[float], maxiter: int = 30, tol: float = 1e-4) -> float:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as e:
        raise ImportError("scipy.optimize module not found") from e

    assert (dim := len(D.shape)) == 4, f"Expected 4-D array, got {dim}-D array"

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

    nc, *_ = D.shape
    grad_f = np.zeros((nc, nc))
    P = random_bistochastic(nc)
    Q = np.zeros((nc, nc))

    for _ in range(maxiter):
        # Compute gradient
        grad_f = np.sum(D * P, axis=(2, 3))

        # Compute the direction Q
        rows, cols = linear_sum_assignment(grad_f)
        Q[:, :] = 0
        Q[rows, cols] = 1

        # Compute the step size alpha
        R = Q - P
        S = np.multiply.outer(R, P)
        a = np.sum(D * np.multiply.outer(R, R))
        b = np.sum(D * (S + np.transpose(S, (2, 3, 0, 1))))

        if a > 0 and 0 <= -b / (2 * a) <= 1:
            alpha = -b / (2 * a)
        else:
            alpha = 0 if a + b > 0 else 1

        # Update P
        P_next = (1 - alpha) * P + alpha * Q
        norm2 = np.sum((P_next - P) ** 2)
        P = P_next

        # Check convergence condition
        if norm2 <= tol**2:
            break

    # Obtain solution
    rows, cols = linear_sum_assignment(P, maximize=True)
    P[:, :] = 0
    P[rows, cols] = 1
    res = np.sum(D * np.multiply.outer(P, P))

    return res


import time
import random
import fastmap
import mapel.elections as mapel
import numpy as np

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


def dist(M_U, M_V, sigma):
    M_V = M_V[sigma, :]
    M_V = M_V[:, sigma]
    return np.sum(np.abs(M_U - M_V))


def bf(M_U: np.ndarray, M_V: np.ndarray):
    from itertools import permutations

    assert M_U.shape == M_V.shape
    nc = M_U.shape[0]

    best = float("inf")
    for sigma in permutations(range(nc)):
        best = min(best, dist(M_U, M_V, sigma))

    return best


nv, nc = 100, 8
culture1 = ORDINAL_CULTURES[random.randint(0, len(ORDINAL_CULTURES) - 1)]
culture2 = ORDINAL_CULTURES[random.randint(0, len(ORDINAL_CULTURES) - 1)]

print(
    "\nPAIRWISE\n\n"
    f"Candidates {nc} :: Votes {nv} :: "
    f"Culture1 {culture1['id']} {culture1['params']} :: "
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


MU, MV = U.votes_to_pairwise_matrix(), V.votes_to_pairwise_matrix()

t1 = time.monotonic()
d1 = bf(MU, MV)
t1 = time.monotonic() - t1
print(f"Python :: {d1:6.3f} :: Time {t1:6.3f}s")

t2 = time.monotonic()
d2 = fastmap.pairwise(
    U.votes_to_pairwise_matrix(),
    V.votes_to_pairwise_matrix(),
    method="faq",
    repeats=30,
    maxiter=300,
    tol=1e-4,
)
t2 = time.monotonic() - t2
print(f"C(faq) :: {d2:6.3f} :: Time {t2:6.3f}s :: Approx. ratio {d2 / d1 if d1 > 0 else d1 == d2:.3f}")

t3 = time.monotonic()
D = np.abs(np.subtract.outer(MU, MV)).swapaxes(1, 2)
d3 = min(qap_faq(D) for _ in range(300))
t3 = time.monotonic() - t3
print(f"P(faq) :: {d3:6.3f} :: Time {t3:6.3f}s :: Approx. ratio {d3 / d1 if d1 > 0 else d1 == d3:.3f}")
