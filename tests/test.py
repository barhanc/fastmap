import unittest

import fastmap
import numpy as np

from typing import Callable
from itertools import permutations


def py_pairwise_bf(M_U: np.ndarray, M_V: np.ndarray) -> int:
    assert isinstance(M_U, np.ndarray) and isinstance(M_V, np.ndarray), "Expected Numpy arrays"
    assert M_U.shape == M_V.shape, "Expected arrays to have the same shape"
    assert (dim := len(M_U.shape)) == 2, f"Expected 2-D arrays, got {dim}-D arrays"
    assert M_U.shape[0] == M_V.shape[1]

    def d(M_U, M_V, sigma):
        M_V = M_V[sigma, :]
        M_V = M_V[:, sigma]
        return np.sum(np.abs(M_U - M_V))

    nc = M_U.shape[0]
    best = float("inf")

    for sigma in permutations(range(nc)):
        best = min(best, d(M_U, M_V, sigma))

    return best


def py_isomorphic_bf(
    U: np.ndarray,
    V: np.ndarray,
    d: Callable[[np.ndarray, np.ndarray], int],
) -> int:
    """Computes Isomorphic d-distance between elections represented by U and V.

    Computes Isomorphic d-distance between elections represented by U and V in a naive way using
    definition
    ```
    d-ID(U, V) := min_{v ∈ S_nv} min_{σ ∈ S_nc} sum_{i=0,..,nv-1} d( U[i, σ], V[v(i), :] )
    ```

    Args:
        U:
            Election matrix. Shape (nv, nc).

            In case of approval election it must be a matrix s.t. U[i,j] ∈ {0,1} is equal to 1 if
            i-th approval ballot in the U election contains j-th candidate and 0 otherwise.

            In case of ordinal election it must be a position matrix s.t. U[i,j] ∈ {0,...,nc-1}
            is equal to the position of j-th candidate in i-th vote.
        V:
            Election matrix. Shape (nv, nc).

            In case of approval election it must be a matrix s.t. V[i,j] ∈ {0,1} is equal to 1 if
            i-th approval ballot in the V election contains j-th candidate and 0 otherwise.

            In case of ordinal election it must be a position matrix s.t. V[i,j] ∈ {0,...,nc-1}
            is equal to the position of j-th candidate in i-th vote.
        d:
            Function to compute the given distance between two votes (e.g. Spearman, Hamming or swap
            distance).

            NOTE: In case of ordinal elections it should take as arguments rows of the position
            matrix, that is vectors which contain positions of subsequent candidates as opposed to
            vectors which contain candidates on subsequent positions.

    Returns:
        Isomorphic d-distance between elections represented by U and V.

    """
    assert isinstance(U, np.ndarray) and isinstance(V, np.ndarray), "Expected Numpy arrays"
    assert U.shape == V.shape, "Expected arrays to have the same shape"
    assert (dim := len(U.shape)) == 2, f"Expected 2-D arrays, got {dim}-D arrays"

    nv, nc = U.shape
    best = float("inf")

    for nu in permutations(range(nv)):
        for sigma in permutations(range(nc)):
            curr = sum(d(U[i, sigma], V[nu[i], :]) for i in range(nv))
            best = min(curr, best)

    return best


def d_spear(u: np.ndarray, v: np.ndarray) -> int:
    return np.sum(np.abs(u - v))


def d_hamm(u: np.ndarray, v: np.ndarray) -> int:
    return np.sum(np.logical_xor(u, v))


def d_swap(u: np.ndarray, v: np.ndarray) -> int:
    return np.sum(np.subtract.outer(u, u) * np.subtract.outer(v, v) < 0) // 2


ORDINAL_CASES = [
    (
        np.array([[2, 0, 1], [0, 2, 1], [2, 1, 0], [2, 0, 1]], dtype=int),
        np.array([[1, 2, 0], [1, 2, 0], [0, 2, 1], [2, 0, 1]], dtype=int),
    ),
    (
        np.array([[2, 1, 0, 3], [0, 1, 2, 3], [3, 0, 1, 2], [2, 0, 1, 3], [0, 2, 1, 3]], dtype=int),
        np.array([[0, 3, 1, 2], [0, 3, 2, 1], [2, 1, 0, 3], [3, 0, 1, 2], [0, 3, 1, 2]], dtype=int),
    ),
    (
        np.array([[0, 1, 2, 3], [1, 0, 3, 2], [1, 0, 2, 3], [0, 1, 3, 2], [3, 2, 1, 0], [1, 0, 2, 3]], dtype=int),
        np.array([[1, 0, 2, 3], [1, 0, 2, 3], [2, 3, 0, 1], [1, 2, 3, 0], [1, 3, 2, 0], [3, 1, 2, 0]], dtype=int),
    ),
]
APPROVAL_CASES = [
    (
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=bool),
        np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=bool),
    ),
    (
        np.array([[0, 1, 0, 1], [0, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=bool),
        np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool),
    ),
    (
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=bool),
        np.array([[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=bool),
    ),
]
PAIRWISE_CASES = [
    (
        np.array([[0.0, 0.75, 0.25], [0.25, 0.0, 0.0], [0.75, 1.0, 0.0]]),
        np.array([[0.0, 0.50, 0.25], [0.50, 0.0, 0.5], [0.75, 0.5, 0.0]]),
    ),
    (
        np.array([[0.0, 0.8, 0.6, 0.8], [0.2, 0.0, 0.4, 0.8], [0.4, 0.6, 0.0, 0.8], [0.2, 0.2, 0.2, 0.0]]),
        np.array([[0.0, 0.8, 0.8, 0.8], [0.2, 0.0, 0.6, 0.2], [0.2, 0.4, 0.0, 0.2], [0.2, 0.8, 0.8, 0.0]]),
    ),
    (
        np.array(
            [
                [0.00, 0.40, 0.22, 0.43, 0.86],
                [0.60, 0.00, 0.54, 0.69, 0.71],
                [0.78, 0.46, 0.00, 0.48, 0.81],
                [0.57, 0.31, 0.52, 0.00, 0.68],
                [0.14, 0.29, 0.19, 0.32, 0.00],
            ]
        ),
        np.array(
            [
                [0.00, 0.09, 0.19, 0.29, 0.53],
                [0.91, 0.00, 0.31, 0.52, 0.77],
                [0.81, 0.69, 0.00, 0.74, 0.91],
                [0.71, 0.48, 0.26, 0.00, 0.94],
                [0.47, 0.23, 0.09, 0.06, 0.00],
            ]
        ),
    ),
]


class TestSpearman(unittest.TestCase):

    correct_values = [py_isomorphic_bf(U.argsort(), V.argsort(), d_spear) for U, V in ORDINAL_CASES]

    def test_bf_correct(self):
        for i in range(len(ORDINAL_CASES)):
            U, V = ORDINAL_CASES[i]
            correct_value = self.correct_values[i]
            with self.subTest(i=i):
                self.assertEqual(
                    (value := fastmap.spearman(U, V, method="bf")),
                    correct_value,
                    f"""Value = {value} returned by 'fastmap.spearman(..., "bf")' method with default options is not equal the correct value = {correct_value}.""",
                )

    def test_bb_correct(self):
        for i in range(len(ORDINAL_CASES)):
            U, V = ORDINAL_CASES[i]
            correct_value = self.correct_values[i]
            with self.subTest(i=i):
                self.assertEqual(
                    (value := fastmap.spearman(U, V, method="bb")),
                    correct_value,
                    f"""Value = {value} returned by 'fastmap.spearman(..., "bb")' method with default options is not equal the correct value = {correct_value}.""",
                )

    def test_aa_geq(self):
        for i in range(len(ORDINAL_CASES)):
            U, V = ORDINAL_CASES[i]
            correct_value = self.correct_values[i]
            with self.subTest(i=i):
                self.assertGreaterEqual(
                    (value := fastmap.spearman(U, V, method="aa")),
                    correct_value,
                    f"""Value = {value} returned by 'fastmap.spearman(..., "aa")' *heuristic* method with default options is not greater or equal the correct value = {correct_value}.""",
                )


class TestHamming(unittest.TestCase):

    correct_values = [py_isomorphic_bf(U, V, d_hamm) for U, V in APPROVAL_CASES]

    def test_bf_correct(self):
        for i in range(len(APPROVAL_CASES)):
            U, V = APPROVAL_CASES[i]
            correct_value = self.correct_values[i]
            with self.subTest(i=i):
                self.assertEqual(
                    (value := fastmap.hamming(U, V, method="bf")),
                    correct_value,
                    f"""Value = {value} returned by 'fastmap.hamming(..., "bf")' method with default options is not equal the correct value = {correct_value}.""",
                )

    def test_bb_correct(self):
        for i in range(len(APPROVAL_CASES)):
            U, V = APPROVAL_CASES[i]
            correct_value = self.correct_values[i]
            with self.subTest(i=i):
                self.assertEqual(
                    (value := fastmap.hamming(U, V, method="bb")),
                    correct_value,
                    f"""Value = {value} returned by 'fastmap.hamming(..., "bb")' method with default options is not equal the correct value = {correct_value}.""",
                )

    def test_aa_geq(self):
        for i in range(len(APPROVAL_CASES)):
            U, V = APPROVAL_CASES[i]
            correct_value = self.correct_values[i]
            with self.subTest(i=i):
                self.assertGreaterEqual(
                    (value := fastmap.hamming(U, V, method="bb")),
                    correct_value,
                    f"""Value = {value} returned by 'fastmap.hamming(..., "aa")' method with default options is not greater or equal the correct value = {correct_value}.""",
                )


class TestSwap(unittest.TestCase):

    correct_values = [py_isomorphic_bf(U.argsort(), V.argsort(), d_swap) for U, V in ORDINAL_CASES]

    def test_bf_correct(self):
        for i in range(len(ORDINAL_CASES)):
            U, V = ORDINAL_CASES[i]
            correct_value = self.correct_values[i]
            with self.subTest(i=i):
                self.assertEqual(
                    (value := fastmap.swap(U, V, method="bf")),
                    correct_value,
                    f"""Value = {value} returned by 'fastmap.swap(..., "bf")' method with default options is not equal the correct value = {correct_value}.""",
                )

    def test_aa_geq(self):
        for i in range(len(ORDINAL_CASES)):
            U, V = ORDINAL_CASES[i]
            correct_value = self.correct_values[i]
            with self.subTest(i=i):
                self.assertGreaterEqual(
                    (value := fastmap.swap(U, V, method="aa")),
                    correct_value,
                    f"""Value = {value} returned by 'fastmap.swap(..., "aa")' method with default options is not greater or equal the correct value = {correct_value}.""",
                )


class TestPairwise(unittest.TestCase):

    correct_values = [py_pairwise_bf(M_U, M_V) for M_U, M_V in PAIRWISE_CASES]

    def test_faq_geq(self):
        for i in range(len(PAIRWISE_CASES)):
            M_U, M_V = PAIRWISE_CASES[i]
            correct_value = self.correct_values[i]
            with self.subTest(i=i):
                self.assertGreaterEqual(
                    (value := round(fastmap.pairwise(M_U, M_V, method="faq"), ndigits=6)),
                    round(correct_value, ndigits=6),
                    f"""Value = {value} returned by 'fastmap.pairwise(..., "faq")' method with default options is not greater or equal the correct value = {correct_value}.""",
                )

    def test_aa_geq(self):
        for i in range(len(PAIRWISE_CASES)):
            M_U, M_V = PAIRWISE_CASES[i]
            correct_value = self.correct_values[i]
            with self.subTest(i=i):
                self.assertGreaterEqual(
                    (value := round(fastmap.pairwise(M_U, M_V, method="aa"), ndigits=6)),
                    round(correct_value, ndigits=6),
                    f"""Value = {value} returned by 'fastmap.pairwise(..., "aa")' method with default options is not greater or equal the correct value = {correct_value}.""",
                )


if __name__ == "__main__":
    unittest.main()
