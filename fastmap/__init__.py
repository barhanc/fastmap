"""
Fastmap is a small Python library that provides optimized C implementations (as well as convenient
Python wrappers) of algorithms computing structural similarity between elections used in the Map of
Elections framework, such as the isomorphic swap and Spearman distances, Hamming distance, pairwise
distance as well as distances based on diversity, agreement and polarization of the elections.
"""

import numpy as np


def spearman(
    U: np.ndarray[int],
    V: np.ndarray[int],
    method: str = "bf",
    repeats: int = 30,
) -> int:
    """Computes Isomorphic Spearman distance between ordinal elections U and V defined as

        min_{v ∈ S_nv} min_{σ ∈ S_nc} sum_{i=0,..,nv-1} sum_{k=0,..,nc-1} d(i,v(i),k,σ(k))

    where d(i,j,k,l) := abs(pos_U[i,k] - pos_V[j,l]), nc is the number of candidates, nv is the
    number of voters, pos_U[i,k] denotes the position of k-th candidate in the i-th vote in the U
    election and S_n denotes the set of all permutations of the set {0,..,n-1}.

    NOTE: This function is a Python wrapper around C extension. For implementation details see
    'pyspear.c' and 'bap.h' files.

    Args:
        U: Ordinal Election matrix s.t. U[i,j] ∈ {0,..,nc-1} is the candidate's number on the j-th
           position in the i-th vote in the U election. Shape (nv, nc).

        V: Ordinal Election matrix s.t. V[i,j] ∈ {0,..,nc-1} is the candidate's number on the j-th
           position in the i-th vote in the V election. Shape (nv, nc).

        method: Method used to compute the distance. Should be one of the

                `"bf"` - uses brute-force to solve the equivalent Bilinear Assignment Problem (BAP).
                    Generates all permutations σ of the set {0,..,min(nv-1,nc-1)} using Heap's
                    algorithm and for each generated permutation σ solves the Linear Assignment
                    Problem (LAP) to obtain the optimal permutation v of {0,..,max(nv-1,nc-1)}. Time
                    complexity of this method is O(min(nv,nc)! * max(nv,nc)**3)

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

                `"bb"` - implements a branch-and-bound algorithm with problem specific branching
                    rules and bounding function which solves the corresponding Bilinear Assignment
                    Problem (BAP). For implementation details see 'bap.h' file.

                    NOTE: This method returns exact value, however the performance may vary greatly
                    depending on the specific problem instance. In general, due to the implemented
                    bounding function, it is recommended that nv ~= nc. In asymmetrical case nv >>
                    nc the "bf" method should provide better performance.

        repeats: Number of times we compute distance using "aa" method (i.e. we sample `repeats`
                 starting permutations and then perform coordinate descent) and choose the smallest
                 value. The "aa" heuristic is also used in the "bb" method for getting initial upper
                 bound. If method is not "aa" or "bb" this option is ignored.

    Returns:
        Isomorphic Spearman distance between U and V.

    Raises:
        ImportError: Raises exception if extension module is not found.
    """
    try:
        import fastmap._spear
    except ImportError as e:
        raise ImportError("C extension module for computing Isomorphic Spearman distance not found") from e

    assert isinstance(U, np.ndarray) and isinstance(V, np.ndarray), "Expected numpy arrays"
    assert U.shape == V.shape, "Expected arrays to have the same shape"
    assert (dim := len(U.shape)) == 2, f"Expected 2-D arrays, got {dim}-D arrays"

    nv, nc = U.shape
    # Transpose matrices if necessary i.e. so that num of rows is min(nv,nc). This makes for a
    # better memory access pattern. We also want to exhaustively search over smaller set of
    # permutations utilising the symmetry between voters and candidates matching and in the C
    # extension implementation we always exhaustively search over rows matching.
    if nv > nc:
        pos_U, pos_V = U.argsort().T, V.argsort().T
    else:
        pos_U, pos_V = U.argsort(), V.argsort()

    return fastmap._spear.spear(
        pos_U.astype(np.int32),
        pos_V.astype(np.int32),
        {"bf": 0, "aa": 1, "bb": 2}[method],
        # Options
        repeats,
    )


def hamming(
    U: np.ndarray[int],
    V: np.ndarray[int],
    method: str = "bf",
    repeats: int = 30,
) -> int:
    """Computes Isomorphic Hamming distance between approval elections U and V defined as

        min_{v ∈ S_nv} min_{σ ∈ S_nc} sum_{i=0,..,nv-1} sum_{k=0,..,nc-1} d(i,v(i),k,σ(k))

    where d(i,j,k,l) := U[i,k] xor V[j,l], nc is the number of candidates, nv is the number of
    voters and S_n denotes the set of all permutations of the set {0,..,n-1}.

    NOTE: This function is a Python wrapper around C extension. For implementation details see
    'pyhamm.c' and 'bap.h' files.

    Args:
        U: Approval Election matrix s.t. U[i,j] ∈ {0,1} is equal to 1 if i-th approval ballot in the
           U election contains j-th candidate and 0 otherwise. Shape (nv, nc).

        V: Approval Election matrix s.t. V[i,j] ∈ {0,1} is equal to 1 if i-th approval ballot in the
           V election contains j-th candidate and 0 otherwise. Shape (nv, nc).

        method: Method used to compute the distance. Should be one of the

                `"bf"` - uses brute-force to solve the equivalent Bilinear Assignment Problem (BAP).
                    Generates all permutations σ of the set {1,..,min(nv,nc)} using Heap's algorithm
                    and for each generated permutation σ solves the Linear Assignment Problem (LAP)
                    to obtain the optimal permutation v of {0,..,max(nv-1,nc-1)}. Time complexity of
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

                `"bb"` - implements a branch-and-bound algorithm with problem specific branching
                    rules and bounding function which solves the corresponding Bilinear Assignment
                    Problem (BAP). For implementation details see 'bap.h' file.

                    NOTE: This method returns exact value, however the performance may vary greatly
                    depending on the specific problem instance. In general, due to the implemented
                    bounding function, it is recommended to use this method if nv ≈ nc. In
                    asymmetrical case nv >> nc the "bf" method should provide better performance.

        repeats: Number of times we compute distance using "aa" method (i.e. we sample `repeats`
                 starting permutations and then perform coordinate descent) and choose the smallest
                 value. The "aa" heuristic is also used in the "bb" method for getting initial upper
                 bound. If method is not "aa" or "bb" this option is ignored.

    Returns:
        Isomorphic Hamming distance between U and V.

    Raises:
        ImportError: Raises exception if extension module is not found.
    """
    try:
        import fastmap._hamm
    except ImportError as e:
        raise ImportError("C extension module for computing Isomorphic Hamming distance not found") from e

    assert isinstance(U, np.ndarray) and isinstance(V, np.ndarray), "Expected numpy arrays"
    assert U.shape == V.shape, "Expected arrays to have the same shape"
    assert (dim := len(U.shape)) == 2, f"Expected 2-D arrays, got {dim}-D arrays"

    nv, nc = U.shape
    # Transpose matrices if necessary i.e. so that num of rows is min(nv,nc). This makes for a
    # better memory access pattern. We also want to exhaustively search over smaller set of
    # permutations utilising the symmetry between voters and candidates matching and in the C
    # extension implementation we always exhaustively search over rows matching.
    if nv > nc:
        U, V = U.T, V.T

    return fastmap._hamm.hamm(
        U.astype(np.int32),
        V.astype(np.int32),
        {"bf": 0, "aa": 1, "bb": 2}[method],
        # Options
        repeats,
    )


def swap(
    U: np.ndarray[int],
    V: np.ndarray[int],
    method: str = "bf",
    repeats: int = 30,
) -> int:
    """
    Computes Isomorphic swap distance between ordinal elections U and V defined as

        min_{v ∈ S_nv} min_{σ ∈ S_nc} sum_{i=0,..,nv-1} sum_{k=0,..,nc-1} sum_{l=0,..,nc-1} d(i,v(i),k,l,σ(k),σ(l))

    where d(i,j,k,l,m,n) := 1/2 * { (pos_U[i,k] - pos_U[i,l]) * (pos_V[j,m] - pos_V[j,n]) < 0 }
    ({...} denoting here the Iverson bracket), nc is the number of candidates, nv is the number of
    voters, pos_U[i,k] denotes the position of k-th candidate in the i-th vote in the U election and
    S_n denotes the set of all permutations of the set {0,..,n-1}.

    NOTE: This function is a Python wrapper around C extension. For implementation details see
    'pyswap.c' file.

    Args:
        U: Ordinal Election matrix s.t. U[i,j] ∈ {0,..,nc-1} is the candidate's number on the j-th
           position in the i-th vote in the U election. Shape (nv, nc).

        V: Ordinal Election matrix s.t. V[i,j] ∈ {0,..,nc-1} is the candidate's number on the j-th
           position in the i-th vote in the V election. Shape (nv, nc).

        method: Method used to compute the distance. Should be one of the

                `"bf"` - uses a brute-force approach which generates all permutations of S_nc using
                    Heap's algorithm and for every generated permutation σ solves the corresponding
                    Linear Assignment Problem (LAP) in order to find the optimal permutation v.
                    Depending on the number of candidates nc it either uses a dynamic approach to
                    compute the cost matrix for LAP or uses a special, precomputed lookup table and
                    dynamically computes the appropriate key to this lookup table to obtain cost
                    matrix. The latter method is more efficient but has large memory requirements
                    thus it is only used for up to nc=10. For details see 'pyswap.c' file.

                `"aa"` - implements a coordinate-descent heuristic, analogous to the Alternating
                    Algorithm for BAP problem (see docs for the fastmap.hamming or fastmap.spearman
                    function). Time complexity of this method is O(N * (nv**3 + nc**3 * nv + nv**2 *
                    nc**2)) where N is the number of iterations it takes for the algorithm to
                    converge.

                    NOTE: This method is much faster in practice than "bf" but there are no
                    theoretical guarantees on the approximation ratio for the used heuristic.

                `"bb"` - TODO:...

        repeats: Number of times we compute distance using "aa" method (i.e. we sample `repeats`
                 starting permutations and then perform coordinate descent) and choose the smallest
                 value. If method is not "aa" this option is ignored.

    Returns:
        Isomorphic swap distance between U and V.

    Raises:
        ImportError: Raises exception if extension module is not found.
    """
    try:
        import fastmap._swap
    except ImportError as e:
        raise ImportError("C extension module for computing Isomorphic swap distance not found") from e

    assert isinstance(U, np.ndarray) and isinstance(V, np.ndarray), "Expected numpy arrays"
    assert U.shape == V.shape, "Expected arrays to have the same shape"
    assert (dim := len(U.shape)) == 2, f"Expected 2-D arrays, got {dim}-D arrays"

    # We transpose the matrices so that the memory access pattern is better
    pos_U, pos_V = U.argsort().T, V.argsort().T

    return fastmap._swap.swap(
        pos_U.astype(np.int32),
        pos_V.astype(np.int32),
        {"bf": 0, "aa": 1}[method],
        # Options
        repeats,
    )


def pairwise(
    M_U: np.ndarray[float],
    M_V: np.ndarray[float],
    method: str = "faq",
    maxiter: int = 30,
    tol: float = 1e-3,
) -> float:
    """Computes pairwise L1 distance between ordinal elections U and V defined as

        min_{σ ∈ S_nc} sum_{i=0,..,nc-1} sum_{j=0,..,nc-1} d(i,σ(i),j,σ(j))

    where d(i,j,k,l) := abs(M_V[i,k] - M_V[j,l]), nc is the number of candidates and S_n denotes the
    set of all permutations of the set {0,..,n-1}. Matrices M_V and M_U are the so called pairwise
    matrices of elections U and V. Pairwise matrix M of an election U is a matrix whose element
    M[i,j] is equal to the number of votes in which i-th candidate comes before the j-th one.

    NOTE: This function is a Python wrapper around C extension. For implementation details see
    'qap.h' and 'pypairwise.c' files.

    Args:
        M_U: Pairwise matrix of ordinal election U. Shape (nc, nc).

        M_V: Pairwise matrix of ordinal election V. Shape (nc, nc).

        method: Method used to compute the distance. Should be one of the

                `"faq"` - implements generalized Fast Approximate QAP (FAQAP) algorithm for
                    aproximately solving general Lawler QAP (as opposed to more restricted Koopmans
                    and Beckmann formulation) described in detail in
                    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0121002.

        maxiter: Integer specifying the max number of Frank-Wolfe iterations performed in the FAQ
                 method of computing the pairwise distance.

        tol: Tolerance for termination in the FAQ method of computing the pairwise distance.
             Frank-Wolfe iteration terminates when ||P_{i} - P_{i+1}||_Frobenius < tol, where P is
             the solution to the relaxed QAP and i is the iteration number.

    Returns:
        Pairwise distance between two elections with pairwise matrices M_U and M_V.

    Raises:
        ImportError: Raises exception if extension module is not found.
    """
    try:
        import fastmap._pairwise
    except ImportError as e:
        raise ImportError("C extension module for computing pairwise distance not found") from e

    assert isinstance(M_U, np.ndarray) and isinstance(M_V, np.ndarray), "Expected numpy arrays"
    assert M_U.shape == M_V.shape, "Expected arrays to have the same shape"
    assert (dim := len(M_U.shape)) == 2, f"Expected 2-D arrays, got {dim}-D arrays"
    assert M_U.shape[0] == M_U.shape[1], "Expected pairwise matrix to be a square matrix"
    assert type(maxiter) == int and maxiter > 0, "Expected `maxiter` to be an int greater than 0"
    assert type(tol) == float and tol > 0, "Expected `tol` to be a float greater than 0"

    return fastmap._pairwise.pairwise(
        M_U.astype(np.double),
        M_V.astype(np.double),
        {"faq": 0}[method],
        # Options
        maxiter,
        tol,
    )
