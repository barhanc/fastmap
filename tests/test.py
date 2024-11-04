import pytest
import fastmap
import numpy as np

from typing import Callable
from itertools import permutations


def py_pairwise_bf(M_U: np.ndarray, M_V: np.ndarray) -> int:
    assert isinstance(M_U, np.ndarray) and isinstance(M_V, np.ndarray), "Expected Numpy arrays"
    assert M_U.shape == M_V.shape, "Expected arrays to have the same shape"
    assert (dim := len(M_U.shape)) == 2, f"Expected 2-D arrays,got {dim}-D arrays"
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
    d-ID(U,V) := min_{v ∈ S_nv} min_{σ ∈ S_nc} sum_{i=0,..,nv-1} d( U[i,σ],V[v(i),:] )
    ```
    where nc is the number of candidates,nv is the number of voters and S_n denotes the set of all
    permutations of the set {0,..,n-1}.

    Args:
        U:
            Election matrix. Shape (nv,nc).

            In case of approval election it must be a matrix s.t. U[i,j] ∈ {0,1} is equal to 1 if
            i-th approval ballot in the U election contains j-th candidate and 0 otherwise.

            In case of ordinal election it must be a position matrix s.t. U[i,j] ∈ {0,...,nc-1}
            is equal to the position of j-th candidate in i-th vote.
        V:
            Election matrix. Shape (nv,nc).

            In case of approval election it must be a matrix s.t. V[i,j] ∈ {0,1} is equal to 1 if
            i-th approval ballot in the V election contains j-th candidate and 0 otherwise.

            In case of ordinal election it must be a position matrix s.t. V[i,j] ∈ {0,...,nc-1}
            is equal to the position of j-th candidate in i-th vote.
        d:
            Function to compute the given distance between two votes (e.g. Spearman,Hamming or swap
            distance).

            NOTE: In case of ordinal elections it should take as arguments rows of the position
            matrix,that is vectors which contain positions of subsequent candidates as opposed to
            vectors which contain candidates on subsequent positions.

    Returns:
        Isomorphic d-distance between elections represented by U and V.

    """
    assert isinstance(U, np.ndarray) and isinstance(V, np.ndarray), "Expected Numpy arrays"
    assert U.shape == V.shape, "Expected arrays to have the same shape"
    assert (dim := len(U.shape)) == 2, f"Expected 2-D arrays,got {dim}-D arrays"

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


# fmt:off
ORDINAL_CASES_SMALL = [
    (
        np.array([[2,0,1],[0,2,1],[2,1,0],[2,0,1]],dtype=int),
        np.array([[1,2,0],[1,2,0],[0,2,1],[2,0,1]],dtype=int),
    ),
    (
        np.array([[2,1,0,3],[0,1,2,3],[3,0,1,2],[2,0,1,3],[0,2,1,3]],dtype=int),
        np.array([[0,3,1,2],[0,3,2,1],[2,1,0,3],[3,0,1,2],[0,3,1,2]],dtype=int),
    ),
    (
        np.array([[0,1,2,3],[1,0,3,2],[1,0,2,3],[0,1,3,2],[3,2,1,0],[1,0,2,3]],dtype=int),
        np.array([[1,0,2,3],[1,0,2,3],[2,3,0,1],[1,2,3,0],[1,3,2,0],[3,1,2,0]],dtype=int),
    ),
]
APPROVAL_CASES_SMALL = [
    (
        np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0]],dtype=bool),
        np.array([[0,1,0],[0,0,0],[0,0,0],[0,0,0]],dtype=bool),
    ),
    (
        np.array([[0,1,0,1],[0,0,1,0],[1,1,0,0],[1,0,0,0],[1,0,0,0]],dtype=bool),
        np.array([[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],dtype=bool),
    ),
    (
        np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0],[0,1,0,0]],dtype=bool),
        np.array([[0,0,0,0],[0,1,1,1],[0,1,1,0],[1,0,0,0],[1,1,1,0],[1,1,1,1]],dtype=bool),
    ),
]
PAIRWISE_CASES_SMALL = [
    (
        np.array([[0.0,0.75,0.25],[0.25,0.0,0.0],[0.75,1.0,0.0]]),
        np.array([[0.0,0.50,0.25],[0.50,0.0,0.5],[0.75,0.5,0.0]]),
    ),
    (
        np.array([[0.0,0.8,0.6,0.8],[0.2,0.0,0.4,0.8],[0.4,0.6,0.0,0.8],[0.2,0.2,0.2,0.0]]),
        np.array([[0.0,0.8,0.8,0.8],[0.2,0.0,0.6,0.2],[0.2,0.4,0.0,0.2],[0.2,0.8,0.8,0.0]]),
    ),
    (
        np.array([[0.00,0.40,0.22,0.43,0.86],[0.60,0.00,0.54,0.69,0.71],[0.78,0.46,0.00,0.48,0.81],[0.57,0.31,0.52,0.00,0.68],[0.14,0.29,0.19,0.32,0.00]]),
        np.array([[0.00,0.09,0.19,0.29,0.53],[0.91,0.00,0.31,0.52,0.77],[0.81,0.69,0.00,0.74,0.91],[0.71,0.48,0.26,0.00,0.94],[0.47,0.23,0.09,0.06,0.00]]),
    ),
]

ORDINAL_CASES_LARGE = [
    (
        np.array([[3,0,2,6,5,4,7,1,8],[5,0,6,4,3,7,2,8,1],[3,7,1,0,4,2,5,6,8],[7,4,3,5,0,2,1,8,6],[1,2,8,0,3,4,5,7,6],[2,7,6,5,1,3,8,4,0],[1,6,3,5,0,8,7,2,4],[4,2,0,7,3,1,5,8,6],[3,6,2,8,1,0,5,4,7]],dtype=int),
        np.array([[4,1,3,5,0,8,2,7,6],[4,2,8,1,0,3,6,5,7],[1,6,3,7,8,4,5,2,0],[5,1,4,2,6,8,3,7,0],[6,0,8,5,2,3,7,4,1],[4,1,5,2,8,3,6,7,0],[0,5,6,3,1,8,4,7,2],[1,5,3,6,0,7,4,2,8],[8,0,2,4,5,6,3,1,7]],dtype=int),
    ),
    (
        np.array([[2,0,4,7,5,1,3,6,8],[8,0,4,2,6,3,7,5,1],[0,2,1,8,3,4,7,6,5],[0,4,5,7,2,3,1,6,8],[0,4,6,8,1,2,5,7,3],[3,1,4,2,8,6,5,7,0],[0,5,8,7,1,6,4,2,3],[0,4,6,2,3,7,8,5,1],[1,2,0,5,7,6,8,3,4],],dtype=int),
        np.array([[0,1,2,3,4,5,6,7,8],[1,2,4,6,7,8,5,3,0],[0,1,2,3,6,7,8,5,4],[0,5,6,7,8,4,3,2,1],[0,1,6,7,8,5,4,3,2],[1,3,4,6,8,7,5,2,0],[1,3,5,7,8,6,4,2,0],[0,1,5,6,8,7,4,3,2],[0,1,2,3,7,8,6,5,4],],dtype=int),
    ),
]

APPROVAL_CASES_LARGE = [
    (
        np.array([[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0,1],[0,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0],[0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0],[0,0,1,0,0,0,1,1,0],],dtype=bool), 
        np.array([[1,0,0,0,0,0,0,0,0],[1,0,0,0,0,1,0,0,0],[1,0,0,0,0,1,1,0,0],[1,1,0,0,0,0,0,1,0],[0,1,0,0,1,0,0,0,1],[1,0,1,0,1,0,0,0,0],[0,0,0,0,1,1,1,0,0],[1,1,0,0,1,0,1,1,0],[1,0,0,0,0,0,0,0,0],],dtype=bool),
    ),
    (
        np.array([[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],],dtype=bool),
        np.array([[1,0,0,0,0,0,1,0,0],[1,0,0,0,0,1,0,0,0],[1,0,1,0,1,0,0,0,0],[0,0,1,1,0,0,1,0,0],[1,1,0,0,1,0,0,1,1],[1,1,0,0,0,0,1,0,0],[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,1,0,1],[1,0,1,1,0,0,0,0,1],],dtype=bool),
    ),
]

PAIRWISE_CASES_LARGE = [
    (
        np.array([[0.000000,0.550000,0.530000,0.380000,0.540000,0.440000,0.510000,0.450000,0.580000],[0.450000,0.000000,0.520000,0.420000,0.430000,0.470000,0.480000,0.370000,0.580000],[0.470000,0.480000,0.000000,0.410000,0.460000,0.440000,0.520000,0.440000,0.520000],[0.620000,0.580000,0.590000,0.000000,0.580000,0.530000,0.580000,0.520000,0.630000],[0.460000,0.570000,0.540000,0.420000,0.000000,0.430000,0.450000,0.430000,0.490000],[0.560000,0.530000,0.560000,0.470000,0.570000,0.000000,0.530000,0.470000,0.580000],[0.490000,0.520000,0.480000,0.420000,0.550000,0.470000,0.000000,0.430000,0.540000],[0.550000,0.630000,0.560000,0.480000,0.570000,0.530000,0.570000,0.000000,0.590000],[0.420000,0.420000,0.480000,0.370000,0.510000,0.420000,0.460000,0.410000,0.000000],]),
        np.array([[0.000000,0.480000,0.410000,0.490000,0.460000,0.480000,0.460000,0.510000,0.450000],[0.520000,0.000000,0.480000,0.500000,0.530000,0.520000,0.520000,0.450000,0.490000],[0.590000,0.520000,0.000000,0.550000,0.500000,0.530000,0.510000,0.560000,0.530000],[0.510000,0.500000,0.450000,0.000000,0.460000,0.490000,0.440000,0.480000,0.450000],[0.540000,0.470000,0.500000,0.540000,0.000000,0.490000,0.540000,0.500000,0.420000],[0.520000,0.480000,0.470000,0.510000,0.510000,0.000000,0.450000,0.510000,0.430000],[0.540000,0.480000,0.490000,0.560000,0.460000,0.550000,0.000000,0.550000,0.420000],[0.490000,0.550000,0.440000,0.520000,0.500000,0.490000,0.450000,0.000000,0.390000],[0.550000,0.510000,0.470000,0.550000,0.580000,0.570000,0.580000,0.610000,0.000000],]),
    ),
    (
        np.array([[0.000000,0.480000,0.490000,0.810000,0.640000,0.510000,0.580000,0.780000,0.570000],[0.520000,0.000000,0.500000,0.720000,0.690000,0.560000,0.630000,0.760000,0.460000],[0.510000,0.500000,0.000000,0.750000,0.630000,0.520000,0.610000,0.690000,0.560000],[0.190000,0.280000,0.250000,0.000000,0.410000,0.350000,0.400000,0.390000,0.420000],[0.360000,0.310000,0.370000,0.590000,0.000000,0.490000,0.460000,0.520000,0.470000],[0.490000,0.440000,0.480000,0.650000,0.510000,0.000000,0.520000,0.700000,0.510000],[0.420000,0.370000,0.390000,0.600000,0.540000,0.480000,0.000000,0.650000,0.500000],[0.220000,0.240000,0.310000,0.610000,0.480000,0.300000,0.350000,0.000000,0.440000],[0.430000,0.540000,0.440000,0.580000,0.530000,0.490000,0.500000,0.560000,0.000000],]),
        np.array([[0.000000,0.450000,0.450000,0.480000,0.480000,0.510000,0.520000,0.550000,0.620000],[0.550000,0.000000,0.490000,0.490000,0.510000,0.540000,0.530000,0.550000,0.600000],[0.550000,0.510000,0.000000,0.510000,0.470000,0.540000,0.560000,0.540000,0.540000],[0.520000,0.510000,0.490000,0.000000,0.510000,0.530000,0.520000,0.500000,0.500000],[0.520000,0.490000,0.530000,0.490000,0.000000,0.570000,0.520000,0.510000,0.500000],[0.490000,0.460000,0.460000,0.470000,0.430000,0.000000,0.520000,0.530000,0.540000],[0.480000,0.470000,0.440000,0.480000,0.480000,0.480000,0.000000,0.500000,0.530000],[0.450000,0.450000,0.460000,0.500000,0.490000,0.470000,0.500000,0.000000,0.490000],[0.380000,0.400000,0.460000,0.500000,0.500000,0.460000,0.470000,0.510000,0.000000],]),
    ),
]

# fmt:on

ORDINAL_CASES_SMALL_CORRECT = {"Swap": [3, 5, 9], "Spearman": [6, 10, 16]}
ORDINAL_CASES_LARGE_CORRECT = {"Swap": [65, 62], "Spearman": [116, 106]}
APPROVAL_CASES_SMALL_CORRECT = {"Hamming": [1, 6, 11]}
APPROVAL_CASES_LARGE_CORRECT = {"Hamming": [12, 25]}
PAIRWISE_CASES_SMALL_CORRECT = {"Pairwise": [1.5, 1.2000000000000002, 3.0999999999999996]}
PAIRWISE_CASES_LARGE_CORRECT = {"Pairwise": [2.34, 5.699999999999999]}


class TestSpearman:

    pyvalue = [py_isomorphic_bf(U.argsort(), V.argsort(), d_spear) for U, V in ORDINAL_CASES_SMALL]

    @pytest.mark.parametrize("i", range(len(ORDINAL_CASES_SMALL)))
    def test_small_bf_correct(self, i: int):
        assert fastmap.spearman(*ORDINAL_CASES_SMALL[i], method="bf") == TestSpearman.pyvalue[i]
        assert TestSpearman.pyvalue[i] == ORDINAL_CASES_SMALL_CORRECT["Spearman"][i]

    @pytest.mark.parametrize("i", range(len(ORDINAL_CASES_SMALL)))
    def test_small_bb_correct(self, i: int):
        assert fastmap.spearman(*ORDINAL_CASES_SMALL[i], method="bb") == TestSpearman.pyvalue[i]
        assert TestSpearman.pyvalue[i] == ORDINAL_CASES_SMALL_CORRECT["Spearman"][i]

    @pytest.mark.parametrize("i", range(len(ORDINAL_CASES_SMALL)))
    def test_small_aa_geq(self, i: int):
        assert (
            fastmap.spearman(*ORDINAL_CASES_SMALL[i], method="aa")
            >= TestSpearman.pyvalue[i]
            == ORDINAL_CASES_SMALL_CORRECT["Spearman"][i]
        )

    @pytest.mark.parametrize("i", range(len(ORDINAL_CASES_LARGE)))
    def test_large_bf_correct(self, i: int):
        assert fastmap.spearman(*ORDINAL_CASES_LARGE[i], method="bf") == ORDINAL_CASES_LARGE_CORRECT["Spearman"][i]

    @pytest.mark.parametrize("i", range(len(ORDINAL_CASES_LARGE)))
    def test_large_bb_correct(self, i: int):
        assert fastmap.spearman(*ORDINAL_CASES_LARGE[i], method="bb") == ORDINAL_CASES_LARGE_CORRECT["Spearman"][i]

    @pytest.mark.parametrize("i", range(len(ORDINAL_CASES_LARGE)))
    def test_large_aa_geq(self, i: int):
        assert fastmap.spearman(*ORDINAL_CASES_LARGE[i], method="aa") >= ORDINAL_CASES_LARGE_CORRECT["Spearman"][i]


class TestHamming:

    pyvalue = [py_isomorphic_bf(U, V, d_hamm) for U, V in APPROVAL_CASES_SMALL]

    @pytest.mark.parametrize("i", range(len(APPROVAL_CASES_SMALL)))
    def test_small_bf_correct(self, i: int):
        assert fastmap.hamming(*APPROVAL_CASES_SMALL[i], method="bf") == TestHamming.pyvalue[i]
        assert TestHamming.pyvalue[i] == APPROVAL_CASES_SMALL_CORRECT["Hamming"][i]

    @pytest.mark.parametrize("i", range(len(APPROVAL_CASES_SMALL)))
    def test_small_bb_correct(self, i: int):
        assert fastmap.hamming(*APPROVAL_CASES_SMALL[i], method="bb") == TestHamming.pyvalue[i]
        assert TestHamming.pyvalue[i] == APPROVAL_CASES_SMALL_CORRECT["Hamming"][i]

    @pytest.mark.parametrize("i", range(len(APPROVAL_CASES_SMALL)))
    def test_small_aa_geq(self, i: int):
        assert fastmap.hamming(*APPROVAL_CASES_SMALL[i], method="bb") >= TestHamming.pyvalue[i]
        assert TestHamming.pyvalue[i] == APPROVAL_CASES_SMALL_CORRECT["Hamming"][i]

    @pytest.mark.parametrize("i", range(len(APPROVAL_CASES_LARGE)))
    def test_large_bf_correct(self, i: int):
        assert fastmap.hamming(*APPROVAL_CASES_LARGE[i], method="bf") == APPROVAL_CASES_LARGE_CORRECT["Hamming"][i]

    @pytest.mark.parametrize("i", range(len(APPROVAL_CASES_LARGE)))
    def test_large_bb_correct(self, i: int):
        assert fastmap.hamming(*APPROVAL_CASES_LARGE[i], method="bb") == APPROVAL_CASES_LARGE_CORRECT["Hamming"][i]

    @pytest.mark.parametrize("i", range(len(APPROVAL_CASES_LARGE)))
    def test_large_aa_geq(self, i: int):
        assert fastmap.hamming(*APPROVAL_CASES_LARGE[i], method="bb") >= APPROVAL_CASES_LARGE_CORRECT["Hamming"][i]


class TestSwap:

    pyvalue = [py_isomorphic_bf(U.argsort(), V.argsort(), d_swap) for U, V in ORDINAL_CASES_SMALL]

    @pytest.mark.parametrize("i", range(len(ORDINAL_CASES_SMALL)))
    def test_small_bf_correct(self, i: int):
        assert fastmap.swap(*ORDINAL_CASES_SMALL[i], method="bf") == TestSwap.pyvalue[i]
        assert TestSwap.pyvalue[i] == ORDINAL_CASES_SMALL_CORRECT["Swap"][i]

    @pytest.mark.parametrize("i", range(len(ORDINAL_CASES_SMALL)))
    def test_small_aa_geq(self, i: int):
        assert fastmap.swap(*ORDINAL_CASES_SMALL[i], method="aa") >= TestSwap.pyvalue[i]
        assert TestSwap.pyvalue[i] == ORDINAL_CASES_SMALL_CORRECT["Swap"][i]

    @pytest.mark.parametrize("i", range(len(ORDINAL_CASES_LARGE)))
    def test_large_bf_correct(self, i: int):
        assert fastmap.swap(*ORDINAL_CASES_LARGE[i], method="bf") == ORDINAL_CASES_LARGE_CORRECT["Swap"][i]

    @pytest.mark.parametrize("i", range(len(ORDINAL_CASES_LARGE)))
    def test_large_aa_geq(self, i: int):
        assert fastmap.swap(*ORDINAL_CASES_LARGE[i], method="aa") >= ORDINAL_CASES_LARGE_CORRECT["Swap"][i]


class TestPairwise:
    pyvalue = [py_pairwise_bf(M_U, M_V) for M_U, M_V in PAIRWISE_CASES_SMALL]

    @pytest.mark.parametrize("i", range(len(PAIRWISE_CASES_SMALL)))
    def test_small_faq_geq(self, i: int):
        assert round(fastmap.pairwise(*PAIRWISE_CASES_SMALL[i], method="faq"), 6) >= round(TestPairwise.pyvalue[i], 6)
        assert round(TestPairwise.pyvalue[i], 6) == round(PAIRWISE_CASES_SMALL_CORRECT["Pairwise"][i], 6)

    @pytest.mark.parametrize("i", range(len(PAIRWISE_CASES_SMALL)))
    def test_small_aa_geq(self, i: int):
        assert round(fastmap.pairwise(*PAIRWISE_CASES_SMALL[i], method="aa"), 6) >= round(TestPairwise.pyvalue[i], 6)
        assert round(TestPairwise.pyvalue[i], 6) == round(PAIRWISE_CASES_SMALL_CORRECT["Pairwise"][i], 6)

    @pytest.mark.parametrize("i", range(len(PAIRWISE_CASES_LARGE)))
    def test_large_faq_geq(self, i: int):
        assert round(fastmap.pairwise(*PAIRWISE_CASES_LARGE[i], method="faq"), 6) >= round(
            PAIRWISE_CASES_LARGE_CORRECT["Pairwise"][i], 6
        )

    @pytest.mark.parametrize("i", range(len(PAIRWISE_CASES_LARGE)))
    def test_large_aa_geq(self, i: int):
        assert round(fastmap.pairwise(*PAIRWISE_CASES_LARGE[i], method="aa"), 6) >= round(
            PAIRWISE_CASES_LARGE_CORRECT["Pairwise"][i], 6
        )
