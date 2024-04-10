import numpy as np


def spearman(U: np.ndarray[int], V: np.ndarray[int], method: str = "bf") -> int:
    """Computes Isomorphic Spearman distance between ordinal elections U and V defined as

        min_{v ∈ S_nv} min_{σ ∈ S_nc} sum_{i=1,..,nv} sum_{k=1,..,nc} d(i,v(i),k,σ(k))

    where d(i,j,k,l) := |pos_U[i,k] - pos_V[j,l]|, nc is the number of candidates, nv is the number
    of voters, pos_U[i,k] denotes the position of k-th candidate in the i-th vote in the U election
    and S_n denotes the set of all permutations of the set {1,..,n}.

    Args:
        U: Ordinal Election matrix s.t. U[i,j] ∈ {1,..,nc} is the candidate's number on the j-th
        position in the i-th voter in the U election. Shape (nv, nc).

        V: Ordinal Election matrix s.t. V[i,j] ∈ {1,..,nc} is the candidate's number on the j-th
        position in the i-th voter in the V election. Shape (nv, nc).

        method: Method used to compute the distance. Should be one of the
                `"bf"` - uses brute-force to solve the equivalent Bilinear Assignment Problem (BAP).
                    Generates all permutations σ of the set {1,..,min(nv,nc)} using Heap's algorithm
                    and for each generated permutation σ solves the Linear Assignment Problem (LAP)
                    to obtain the optimal permutation v of {1,..,max(nv,nc)}. Time complexity of
                    this method is O(min(nv,nc)! * max(nv,nc)^3)

                    NOTE: This method returns exact value but if one of the nv, nc is greater than
                    10 it is extremely slow.

                `"aa"` - implements Alternating Algorithm heuristic described in arXiv:1707.07057
                    which solves the equivalent Bilinear Assignment Problem (BAP). The algorithm
                    first generates a feasible solution to the BAP using (TODO: Choose
                    initialization method) and then performs a coordinate-descent-like refinment by
                    solving Linear Assignment Problem (LAP) with one of the permutations fixed until
                    convergence.

                    NOTE: This method is much faster than "bf" but there are no theoretical
                    guarantees on approximation ratio for the used heuristic.

    Returns:
        Isomorphic Spearman distance between U and V.
    """
    import fastmap._spear

    assert U.shape == V.shape, "Expected arrays to have the same shape"
    assert (dim := len(U.shape)) == 2, f"Expected 2D arrays, got {dim}D arrays"

    methods = {"bf": 0, "aa": 1}

    nv, nc = U.shape
    if nv < nc:
        pos_U, pos_V = U.argsort().T, V.argsort().T
    else:
        pos_U, pos_V = U.argsort(), V.argsort()
    pos_U, pos_V = pos_U.astype(np.int32), pos_V.astype(np.int32)

    return fastmap._spear.spear(pos_U, pos_V, methods[method])


def hamming(U: np.ndarray[int], V: np.ndarray[int], method: str = "bf") -> int:
    """Computes Isomorphic Hamming distance between approval elections U and V defined as

        min_{v ∈ S_nv} min_{σ ∈ S_nc} sum_{i=1,..,nv} sum_{k=1,..,nc} d(i,v(i),k,σ(k))

    where d(i,j,k,l) := U[i,k] xor V[j,l], nc is the number of candidates, nv is the number of
    voters and S_n denotes the set of all permutations of the set {1,..,n}.

    Args:
        U: Approval Election matrix s.t. U[i,j] ∈ {0,1} is equal to 1 if i-th approval ballot in the
        U election contains j-th candidate and 0 otherwise. Shape (nv, nc).

        V: Approval Election matrix s.t. V[i,j] ∈ {0,1} is equal to 1 if i-th approval ballot in the
        V election contains j-th candidate and 0 otherwise. Shape (nv, nc).

        method: Method used to compute the distance. Should be one of the
                `"bf"` - uses brute-force to solve the equivalent Bilinear Assignment Problem (BAP).
                    Generates all permutations σ of the set {1,..,min(nv,nc)} using Heap's algorithm
                    and for each generated permutation σ solves the Linear Assignment Problem (LAP)
                    to obtain the optimal permutation v of {1,..,max(nv,nc)}. Time complexity of
                    this method is O(min(nv,nc)! * max(nv,nc)^3)

                    NOTE: This method returns exact value but if one of the nv, nc is greater than
                    10 it is extremely slow.

                `"aa"` - implements Alternating Algorithm heuristic described in arXiv:1707.07057
                    which solves the equivalent Bilinear Assignment Problem (BAP). The algorithm
                    first generates a feasible solution to the BAP using (TODO: Choose
                    initialization method) and then performs a coordinate-descent-like refinment by
                    solving Linear Assignment Problem (LAP) with one of the permutations fixed until
                    convergence.

                    NOTE: This method is much faster than "bf" but there are no theoretical
                    guarantees on approximation ratio for the used heuristic.

    Returns:
        Isomorphic Hamming distance between U and V.
    """
    import fastmap._hamm

    assert U.shape == V.shape, "Expected arrays to have the same shape"
    assert (dim := len(U.shape)) == 2, f"Expected 2D arrays, got {dim}D arrays"

    methods = {"bf": 0, "aa": 1}

    nv, nc = U.shape
    if nv < nc:
        U, V = U.T, V.T
    U, V = U.astype(np.int32), V.astype(np.int32)

    return fastmap._hamm.hamm(U, V, methods[method])