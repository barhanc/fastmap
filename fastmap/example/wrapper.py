import numpy as np

import fastmap.cspear
import fastmap.chamm


def spearman(U: np.ndarray[int], V: np.ndarray[int]) -> int:
    """Computes Isomorphic Spearman distance between ordinal elections U and V defined as

        min_{v ∈ S_nv} min_{σ ∈ S_nc} sum_{i=1,..,nv} sum_{k=1,..,nc} d(i,v(i),k,σ(k))

    where d(i,j,k,l) := |pos_U[i,k] - pos_V[j,l]|, nc is the number of candidates, nv is the number
    of voters and pos_U[i,k] denotes the position of k-th candidate in the i-th vote in the U
    election.

    Args:
        U: Ordinal Election matrix s.t. U[i,j] ∈ {1,..,nc} is the candidate's number on the j-th
        position in the i-th voter in the U election. Shape (nv, nc).

        V: Ordinal Election matrix s.t. V[i,j] ∈ {1,..,nc} is the candidate's number on the j-th
        position in the i-th voter in the V election. Shape (nv, nc).

    Returns:
        Isomorphic Spearman distance between U and V.
    """
    assert U.shape == V.shape, "Expected arrays to have the same shape"
    assert (dim := len(U.shape)) == 2, f"Expected 2D arrays, got {dim}D arrays"

    nv, nc = U.shape
    pos_U, pos_V = U.argsort().T if nv < nc else U.argsort(), V.argsort().T if nv < nc else V.argsort()
    pos_U, pos_V = pos_U.astype(np.int32), pos_V.astype(np.int32)

    res = fastmap.cspear.spear(pos_U, pos_V)
    if res < 0:
        raise RuntimeError("Error ocurred while using C extension")
    return res


def hamming(U: np.ndarray[int], V: np.ndarray[int]) -> int:
    """Computes Isomorphic Hamming distance between approval elections U and V defined as

        min_{v ∈ S_nv} min_{σ ∈ S_nc} sum_{i=1,..,nv} sum_{k=1,..,nc} d(i,v(i),k,σ(k))

    where d(i,j,k,l) := U[i,k] xor V[j,l], nc is the number of candidates and nv is the number of
    voters.

    Args:
        U: Approval Election matrix s.t. U[i,j] ∈ {0,1} is equal to 1 if i-th approval ballot in the
        U election contains j-th candidate and 0 otherwise. Shape (nv, nc).

        V: Approval Election matrix s.t. V[i,j] ∈ {0,1} is equal to 1 if i-th approval ballot in the
        V election contains j-th candidate and 0 otherwise. Shape (nv, nc).

    Returns:
        Isomorphic Hamming distance between U and V.
    """
    assert U.shape == V.shape, "Expected arrays to have the same shape"
    assert (dim := len(U.shape)) == 2, f"Expected 2D arrays, got {dim}D arrays"

    U, V = U.astype(np.int32), V.astype(np.int32)

    res = fastmap.chamm.hamm(U, V)
    if res < 0:
        raise RuntimeError("Error ocurred while using C extension")
    return res
