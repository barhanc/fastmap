import numpy as np


def spearman(U: np.ndarray[int], V: np.ndarray[int], method: str = "bf") -> int:
    """Computes Isomorphic Spearman distance between ordinal elections U and V defined as

        min_{v ∈ S_nv} min_{σ ∈ S_nc} sum_{i=1,..,nv} sum_{k=1,..,nc} d(i,v(i),k,σ(k))

    where d(i,j,k,l) := |pos_U[i,k] - pos_V[j,l]|, nc is the number of candidates, nv is the number
    of voters, pos_U[i,k] denotes the position of k-th candidate in the i-th vote in the U election
    and S_n denotes the set of all permutations of the set {1,..,n}.

    Args:
        U: Ordinal Election matrix s.t. U[i,j] ∈ {1,..,nc} is the candidate's number on the j-th
        position in the i-th vote in the U election. Shape (nv, nc).

        V: Ordinal Election matrix s.t. V[i,j] ∈ {1,..,nc} is the candidate's number on the j-th
        position in the i-th vote in the V election. Shape (nv, nc).

        method: Method used to compute the distance. Should be one of the
                `"bf"` - uses brute-force to solve the equivalent Bilinear Assignment Problem (BAP).
                    Generates all permutations σ of the set {1,..,min(nv,nc)} using Heap's algorithm
                    and for each generated permutation σ solves the Linear Assignment Problem (LAP)
                    to obtain the optimal permutation v of {1,..,max(nv,nc)}. Time complexity of
                    this method is O(min(nv,nc)! * max(nv,nc)**3)

                    NOTE: This method returns exact value but if one of the nv, nc is greater than
                    10 it is extremely slow.

                `"aa"` - implements Alternating Algorithm heuristic described in arXiv:1707.07057
                    which solves the equivalent Bilinear Assignment Problem (BAP). The algorithm
                    first generates a feasible solution to the BAP by sampling from a uniform
                    distribution two permutations σ, v and then performs a coordinate-descent-like
                    refinment by interchangeably fixing one of the permutations, solving the
                    corresponding Linear Assignment Problem (LAP) and updating the other permutation
                    with the matching found in LAP, doing so until convergence. Time complexity of
                    this method is O(N * (nv**3 + nc**3)) where N is the number of iterations it
                    takes for the algorithm to converge.

                    NOTE: This method is much faster in practice than "bf" but there are no
                    theoretical guarantees on the approximation ratio for the used heuristic.

    Returns:
        Isomorphic Spearman distance between U and V.
    """
    import fastmap._spear

    assert isinstance(U, np.ndarray) and isinstance(V, np.ndarray), "Expected numpy arrays"
    assert U.shape == V.shape, "Expected arrays to have the same shape"
    assert (dim := len(U.shape)) == 2, f"Expected 2-D arrays, got {dim}-D arrays"

    nv, nc = U.shape
    if nv < nc:
        pos_U, pos_V = U.argsort().T, V.argsort().T
    else:
        pos_U, pos_V = U.argsort(), V.argsort()

    return fastmap._spear.spear(
        pos_U.astype(np.int64),
        pos_V.astype(np.int64),
        {"bf": 0, "aa": 1}[method],
    )


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
                    this method is O(min(nv,nc)! * max(nv,nc)**3)

                    NOTE: This method returns exact value but if one of the nv, nc is greater than
                    10 it is extremely slow.

                `"aa"` - implements Alternating Algorithm heuristic described in arXiv:1707.07057
                    which solves the equivalent Bilinear Assignment Problem (BAP). The algorithm
                    first generates a feasible solution to the BAP by sampling from a uniform
                    distribution two permutations σ, v and then performs a coordinate-descent-like
                    refinment by interchangeably fixing one of the permutations, solving the
                    corresponding Linear Assignment Problem (LAP) and updating the other permutation
                    with the matching found in LAP, doing so until convergence. Time complexity of
                    this method is O(N * (nv**3 + nc**3)) where N is the number of iterations it
                    takes for the algorithm to converge.

                    NOTE: This method is much faster in practice than "bf" but there are no
                    theoretical guarantees on the approximation ratio for the used heuristic.

    Returns:
        Isomorphic Hamming distance between U and V.
    """
    import fastmap._hamm

    assert isinstance(U, np.ndarray) and isinstance(V, np.ndarray), "Expected numpy arrays"
    assert U.shape == V.shape, "Expected arrays to have the same shape"
    assert (dim := len(U.shape)) == 2, f"Expected 2-D arrays, got {dim}-D arrays"

    nv, nc = U.shape
    if nv < nc:
        U, V = U.T, V.T

    return fastmap._hamm.hamm(
        U.astype(np.int64),
        V.astype(np.int64),
        {"bf": 0, "aa": 1}[method],
    )

def swap(U: np.ndarray[int], V: np.ndarray[int], method: str = "bf") -> int:
    """Computes Isomorphic Swap distance between approval elections U and V defined as

        idk_lmao_lol_todoiguess_schizo

    Args:
        U: Approval Election matrix s.t. U[i,j] ∈ {0,1} is equal to 1 if i-th approval ballot in the
        U election contains j-th candidate and 0 otherwise. Shape (nv, nc).

        V: Approval Election matrix s.t. V[i,j] ∈ {0,1} is equal to 1 if i-th approval ballot in the
        V election contains j-th candidate and 0 otherwise. Shape (nv, nc).

        method: Method used to compute the distance. Should be one of the
                `"bf"` -  brute force. Hard limit for 10 votes/candidates.
                `"aa"` - not implemented yet.
    Returns:
        Isomorphic Swap distance between U and V.
    """
    import fastmap._swap

    assert isinstance(U, np.ndarray) and isinstance(V, np.ndarray), "Expected numpy arrays"
    assert U.shape == V.shape, "Expected arrays to have the same shape"
    assert (dim := len(U.shape)) == 2, f"Expected 2-D arrays, got {dim}-D arrays"

    nv, nc = U.shape
    # if nv > nc:
    #     pos_U, pos_V = U.argsort().T, V.argsort().T
    # else:
    #     pos_U, pos_V = U.argsort().T, V.argsort().T

    return fastmap._swap.swap(
        U.astype(np.int64),
        V.astype(np.int64),
        {"bf": 0, "aa": 1}[method],
    )
