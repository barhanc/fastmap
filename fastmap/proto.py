import time

from typing import Callable
from itertools import permutations

import numpy as np
import cvxpy as cp

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def dspear(u: np.ndarray[int], v: np.ndarray[int]):
    return np.sum(np.abs(u.argsort() - v.argsort()))


def dhamming(u: np.ndarray[int], v: np.ndarray[int]):
    return np.sum(np.not_equal(u.argsort(), v.argsort()))


def dswap(u: np.ndarray[int], v: np.ndarray[int]):
    pos_u, pos_v = u.argsort(), v.argsort()
    return np.sum(np.sign(np.subtract.outer(pos_u, pos_u)) != np.sign(np.subtract.outer(pos_v, pos_v))) // 2


def bf(
    U: np.ndarray[int],
    V: np.ndarray[int],
    d: Callable[[np.ndarray[int], np.ndarray[int]], int | float],
) -> int | float:
    assert U.shape == V.shape
    n_votes, n_cands = U.shape[0], U.shape[1]
    best_res = float("inf")

    for nu in permutations(range(n_votes)):
        for sigma in permutations(range(n_cands)):
            sigma = np.array(sigma)
            res = sum(d(sigma[U[i]], V[nu[i]]) for i in range(n_votes))
            best_res = min(res, best_res)

    return best_res


def bf_with_cand_match(
    U: np.ndarray[int],
    V: np.ndarray[int],
    d: np.ndarray[int] | Callable[[np.ndarray[int], np.ndarray[int]], int | float] | str,
) -> int | float:
    assert U.shape == V.shape

    n_cands = U.shape[1]
    best_res = float("inf")
    identity = np.arange(n_cands)

    for sigma in permutations(range(n_cands)):
        sigma = np.array(sigma)
        cost = d[:, :, identity, sigma].sum(-1) if isinstance(d, np.ndarray) else cdist(sigma[U], V, metric=d)
        row, col = linear_sum_assignment(cost)
        best_res = min(cost[row, col].sum(), best_res)

    return best_res


def spear_ilp(U: np.ndarray[int], V: np.ndarray[int]) -> float:
    assert U.shape == V.shape
    n_votes, n_cands = U.shape[0], U.shape[1]
    pos_U = np.argsort(U)
    pos_V = np.argsort(V)

    D = np.abs(np.subtract.outer(pos_U, pos_V)).reshape(n_votes * n_cands, n_votes * n_cands, order="F")
    N = cp.Variable((n_votes, n_votes), boolean=True)
    M = cp.Variable((n_cands, n_cands), boolean=True)
    P = cp.Variable((n_votes * n_cands, n_votes * n_cands), boolean=True)

    I_N = np.ones(n_votes)
    I_M = np.ones(n_cands)
    I_P = np.ones(n_cands * n_votes)
    I_NN = np.ones((n_votes, n_votes))
    I_MM = np.ones((n_cands, n_cands))

    constraints = [
        N @ I_N == 1,
        I_N @ N == 1,
        M @ I_M == 1,
        I_M @ M == 1,
        P @ I_P == 1,
        I_P @ P == 1,
        P <= cp.kron(I_MM, N),
        P <= cp.kron(M, I_NN),
    ]

    prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(P, D))), constraints)
    res = prob.solve(solver="CPLEX", verbose=False)

    return res


def spear_ilp2(U: np.ndarray, V: np.ndarray) -> float:
    """
    Based on "An algorithm for the quadratic assignment problem using Benders' decomposition",
    L.Kaufman, F.Broeckx
    """
    assert U.shape == V.shape
    n_votes, n_cands = U.shape[0], U.shape[1]
    pos_U = np.argsort(U)
    pos_V = np.argsort(V)

    D = np.abs(np.subtract.outer(pos_U, pos_V)).swapaxes(1, 2).reshape(n_votes**2, n_cands**2, order="F")
    N = cp.Variable((n_votes, n_votes), boolean=True)
    M = cp.Variable((n_cands, n_cands), boolean=True)
    W = cp.Variable((n_votes, n_votes), nonneg=True)

    I_N = np.ones(n_votes)
    I_M = np.ones(n_cands)

    A = (D @ M.reshape(n_cands**2)).reshape((n_votes, n_votes)).T
    B = (D @ np.ones(n_cands**2)).reshape((n_votes, n_votes))

    constraints = [
        N @ I_N == 1,
        I_N @ N == 1,
        M @ I_M == 1,
        I_M @ M == 1,
        W >= A - cp.multiply(B, 1 - N),
    ]

    prob = cp.Problem(cp.Minimize(W.sum()), constraints)
    res = prob.solve(solver="CPLEX", verbose=True)

    return res


import mapel.elections.distances.cppdistances as dist

if __name__ == "__main__":
    n_votes, n_cands = 8, 8
    V1 = np.array([np.random.permutation(n_cands) for _ in range(n_votes)])
    V2 = np.array([np.random.permutation(n_cands) for _ in range(n_votes)])

    # s = time.perf_counter()
    # res = bf(V1, V2, dspear)
    # e = time.perf_counter()
    # print(f"BF: {res}, {e-s:.4f}")

    s = time.perf_counter()
    pos1 = np.argsort(V1)
    pos2 = np.argsort(V2)
    d = np.abs(np.subtract.outer(pos1, pos2)).swapaxes(1, 2)  # Spearman
    # d = np.not_equal.outer(pos1, pos2).swapaxes(1, 2)  # Hamming
    # d = dswap
    # d = dspear
    res = bf_with_cand_match(V1, V2, d)
    e = time.perf_counter()
    print(f"BF with Cand. Match Precomputed D: {res}, {e-s:.4f}")

    s = time.perf_counter()
    res = dist.speard(V1, V2)
    # res = dist.swapd(V1, V2)
    e = time.perf_counter()
    print(f"Cpp TO BEAT (with no extensions) : {res}, {e-s:.4f}")

    # s = time.perf_counter()
    # res = spear_ilp(V1, V2)
    # e = time.perf_counter()
    # print(f"ILP form. 1:                     {res}, {e-s:.4f}")

    # s = time.perf_counter()
    # res = spear_ilp2(V1, V2)
    # e = time.perf_counter()
    # print(f"ILP form. 2:                     {res}, {e-s:.4f}")
