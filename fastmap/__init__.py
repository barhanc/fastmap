"""
Fastmap provides optimized algorithms implementation for Maps of Elections framework.

Fastmap is a small Python library that provides optimized C implementations (as well as convenient
Python wrappers) of algorithms computing structural similarity between elections used in the Map of
Elections framework, such as the isomorphic swap and Spearman distances, Hamming distance, pairwise
distance as well as distances based on diversity, agreement and polarization of the elections.
"""

import ctypes
import numpy as np
from mapel.elections.objects.Election import Election


def agreement_index(election: Election) -> dict[str, float]:
    """Calculates the Agreement Index for a voting matrix.

    The Agreement Index quantifies the level of consensus among voters for a set of candidates
    in an election. The index is derived by computing the candidate dominance distances and
    averaging over all candidate pairs.

    Args:
        election:
            Election object from mapel library.

    Returns:
        A float representing the Agreement Index, normalized by the number of candidate pairs.

    Raises:
        ImportError: Raises exception if C extension module for Agreement Index is not found.
        MemoryError: If memory allocation for distances fails.
    """
    try:
        import fastmap._agreement_index
    except ImportError as e:
        raise ImportError("Error while importing C extension for computing Agreement Index") from e
    
    if election.fake:
        return {'value': None}

    assert isinstance(election, Election), "Expected an Election"

    return {'value': fastmap._agreement_index.agreement_index(election.votes)}

def polarization_index(election: Election) -> dict[str, float]:
    """Calculates the Polarization Index for a voting matrix.

    The Polarization Index quantifies the level of polarization among voters for a set of candidates
    in an election. The index is derived from the analysis of voter distances and their distribution.

    Args:
        election:
            Election object from mapel library.

    Returns:
        A float representing the Polarization Index, normalized by the number of candidate pairs.

    Raises:
        ImportError: Raises exception if C extension module for Polarization Index is not found.
    """
    try:
        import fastmap._polarization_index
    except ImportError as e:
        raise ImportError("Error while importing C extension for computing Polarization Index") from e

    assert isinstance(election, Election), "Expected an Election"

    return {'value': fastmap._polarization_index.polarization_index(election.votes)}

def diversity_index(election: Election) -> dict[str, float]:
    """Calculates the Diversity Index for a voting matrix.

    The Diversity Index quantifies the level of diversity among voters for a set of candidates
    in an election. The index is derived from the analysis of voter distances and their distribution.

    Args:
        election:
            Election object from mapel library.

    Returns:
        A float representing the Diversity Index, normalized by the number of candidate pairs.

    Raises:
        ImportError: Raises exception if C extension module for Diversity Index is not found.
        ValueError: If the input votes array has an incompatible shape or invalid data.
    """

    try:
        import fastmap._diversity_index
    except ImportError as e:
        raise ImportError("Error while importing C extension for computing Diversity Index") from e

    assert isinstance(election, Election), "Expected an Election object"

    return {'value': fastmap._diversity_index.diversity_index(election.votes)}


def kemeny_ranking(election: Election) -> tuple[np.ndarray, float]:
    """Calculates the Kemeny ranking for a voting matrix.

    The Kemeny ranking identifies the permutation of candidates that minimizes the distance to
    all voters' rankings, using a pairwise comparison approach.

    Args:
        election:
            Election object from mapel library.

    Returns:
        A tuple containing:
        - A numpy array with the best ranking (an ordered list of candidate indices).
        - A float representing the best distance achieved by this ranking.

    Raises:
        ImportError: If the C extension module for Kemeny ranking is not found.
        ValueError: If the input votes array has an incompatible shape or invalid data.
        MemoryError: If memory allocation for intermediate matrices fails.
    """
    try:
        from ._kemeny_ranking import kemeny_ranking as _kemeny_ranking
    except ImportError as e:
        raise ImportError("Error while importing C extension for computing Kemeny ranking") from e

    if not isinstance(election, Election):
        raise ValueError("Expected a numpy array for votes")

    return _kemeny_ranking(election.votes)


def local_search_kKemeny_single_k(election: Election, k: int, l: int, starting: np.ndarray[int] | None = None) -> dict[str, int]:
    """Performs local search for Kemeny ranking optimization using a single k.

    Args:
        election:
            Election object from mapel library.
        k:
            The number of candidates to consider in the local search.
        l:
            A parameter controlling the local search's stopping criteria.
        starting:
            An optional 1-D numpy array of integers indicating the starting ranking (default is None).

    Returns:
        An integer representing the resulting score of the Kemeny optimization.

    Raises:
        ImportError: Raises exception if C extension module for Kemeny local search is not found.
        ValueError: If the input votes array has an incompatible shape or invalid data.
    """

    try:
        import fastmap._local_search_kkemeny_single_k
    except ImportError as e:
        raise ImportError("Error while importing C extension for computing local search Kemeny") from e

    assert isinstance(election, Election), "Expected an Election object"

    if starting is not None:
        assert isinstance(starting, np.ndarray), "Starting should be a numpy array"
        assert starting.ndim == 1 and starting.size == k, "Starting array must be 1-D with size k"
        assert starting.dtype == np.int32, "Expected starting array of dtype np.int32"
        starting_ptr = starting.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    else:
        starting_ptr = []

    result = fastmap._local_search_kkemeny_single_k.local_search_kkemeny_single_k(
        election.votes, k, l, election.num_voters, election.num_candidates, starting_ptr
    )
    
    return {'value': result}

def local_search_kKemeny(
    election: Election,
    l: int,
    starting: np.ndarray[int] = None
) -> dict[str, np.ndarray]:
    """Performs local search for Kemeny ranking optimization on a set of votes.

    Args:
        election:
            Election object from mapel library.
        l:
            A parameter controlling the local search's stopping criteria.
        starting:
            An optional 1-D numpy array of integers indicating the starting ranking (default is None).

    Returns:
        A 1-D numpy array of floats representing the resulting scores for each voter.

    Raises:
        ImportError: If the C extension module for Kemeny local search is not found.
        ValueError: If the input votes array has an incompatible shape or invalid data.
    """
    try:
        import fastmap._local_search_kkemeny  # Ensure to replace with the actual module name
    except ImportError as e:
        raise ImportError("Error while importing C extension for computing local search Kemeny") from e

    assert isinstance(election, Election), "Expected an Election"
    
    if starting is not None:
        assert isinstance(starting, np.ndarray), "Starting should be a numpy array"
        assert starting.ndim == 1 and starting.size == election.num_candidates, \
            "Starting array must be 1-D with size equal to number of candidates"
        assert starting.dtype == np.int32, "Expected starting array of dtype np.int32"
        starting_ptr = starting.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    else:
        starting_ptr = []
    
    result = fastmap._local_search_kkemeny.local_search_kkemeny(
        election.votes, l, election.num_voters, election.num_candidates, starting_ptr
    )

    return {'value': np.array(result, dtype=np.float64)}


def polarization_1by2Kemenys(election: Election) -> float:
    """Calculates polarization between two Kemeny rankings.

    Args:
        election:
            Election object from mapel library.

    Returns:
        A float representing the polarization score between two Kemeny rankings.

    Raises:
        ImportError: If the C extension for polarization computation is not found.
        ValueError: If the input votes array has an incompatible shape or invalid data.
    """
    try:
        import fastmap._polarization_1by2Kemenys as _polarization_1by2Kemenys
    except ImportError as e:
        raise ImportError("Error importing the C extension for computing polarization") from e

    assert isinstance(election, Election), "Expected an Election"

    return {'value': _polarization_1by2Kemenys.polarization_1by2Kemenys(election.votes)}


def greedy_kmeans_summed(election: Election) -> dict[str, float]:
    """Calculates a greedy K-means summed score from vote data.

    Args:
        election:
            Election object from mapel library.

    Returns:
        A float representing the summed score from the greedy K-means calculation.

    Raises:
        ImportError: If the C extension for the computation is not found.
        ValueError: If the input votes array has an incompatible shape or invalid data.
    """
    try:
        import fastmap._greedy_kmeans_summed as _greedy_kmeans  # Replace with actual module path if necessary
    except ImportError as e:
        raise ImportError("Error importing the C extension for greedy K-means computation") from e

    assert isinstance(election, Election), "Expected an Election"

    return {'value': _greedy_kmeans.greedy_kmeans_summed(election.votes)}


def greedy_kKemenys_summed(election: Election) -> dict[str, float]:
    """Calculates a greedy Kemeny's summed score from vote data.

    Args:
        election:
            Election object from mapel library.

    Returns:
        A float representing the summed score from the greedy Kemeny's calculation.

    Raises:
        ImportError: If the C extension for the computation is not found.
        ValueError: If the input votes array has an incompatible shape or invalid data.
    """
    try:
        import fastmap._greedy_kKemenys_summed as _greedy_kKemenys
    except ImportError as e:
        raise ImportError("Error importing the C extension for greedy Kemeny's computation") from e

    assert isinstance(election, Election), "Expected an Election"

    return {'value': _greedy_kKemenys.greedy_kKemenys_summed(election.votes)}


def greedy_2kKemenys_summed(election: Election) -> dict[str, float]:
    """Calculates a greedy 2-Kemeny's summed score from vote data.

    Args:
        election:
            Election object from mapel library.

    Returns:
        A float representing the summed score from the greedy 2-Kemeny's calculation.

    Raises:
        ImportError: If the C extension for the computation is not found.
        ValueError: If the input votes array has an incompatible shape or invalid data.
    """
    try:
        import fastmap._greedy_2kKemenys_summed as _greedy_2kKemenys
    except ImportError as e:
        raise ImportError("Error importing the C extension for greedy 2-Kemeny's computation") from e

    assert isinstance(election, Election), "Expected an Election"

    return {'value': _greedy_2kKemenys.greedy_2kKemenys_summed(election.votes)}


def greedy_kKemenys_divk_summed(election: Election) -> dict[str, float]:
    """Calculates a greedy Kemeny's div-k summed score from vote data.

    Args:
        election:
            Election object from mapel library.

    Returns:
        A float representing the summed score from the greedy Kemeny's div-k calculation.

    Raises:
        ImportError: If the C extension for the computation is not found.
        ValueError: If the input votes array has an incompatible shape or invalid data.
    """
    try:
        import fastmap._greedy_kKemenys_divk_summed as _greedy_kKemenys_divk
    except ImportError as e:
        raise ImportError("Error importing the C extension for greedy Kemeny's div-k computation") from e

    assert isinstance(election, Election), "Expected an Election"

    return {'value': _greedy_kKemenys_divk.greedy_kKemenys_divk_summed(election.votes)}


def spearman(
    U: np.ndarray[int],
    V: np.ndarray[int],
    method: str = "bf",
    repeats: int = 30,
    seed: int = -1,
) -> int:
    """Computes Isomorphic Spearman distance between ordinal elections U and V.

    Computes Isomorphic Spearman distance between ordinal elections U and V defined as
    ```
    min_{v ∈ S_nv} min_{σ ∈ S_nc} sum_{i=0,..,nv-1} sum_{k=0,..,nc-1} d(i,v(i),k,σ(k))
    ```
    where d(i,j,k,l) := abs(pos_U[i,k] - pos_V[j,l]), nc is the number of candidates, nv is the
    number of voters, pos_U[i,k] denotes the position of k-th candidate in the i-th vote in the U
    election and S_n denotes the set of all permutations of the set {0,..,n-1}.

    NOTE: This function is a Python wrapper around C extension. For implementation details see
    'pyspear.c' and 'bap.h' files.

    Args:
        U:
            Ordinal Election matrix s.t. U[i,j] ∈ {0,..,nc-1} is the candidate's number on the j-th
            position in the i-th vote in the U election. Shape (nv, nc).

        V:
            Ordinal Election matrix s.t. V[i,j] ∈ {0,..,nc-1} is the candidate's number on the j-th
            position in the i-th vote in the V election. Shape (nv, nc).

        method:
            Method used to compute the distance. Should be one of the

            `"bf"` - uses brute-force to solve the equivalent Bilinear Assignment Problem (BAP).
            Generates all permutations σ of the set {0,..,min(nv-1,nc-1)} using Heap's algorithm and
            for each generated permutation σ solves the Linear Assignment Problem (LAP) to obtain
            the optimal permutation v of {0,..,max(nv-1,nc-1)}. Time complexity of this method is
            `O(min(nv,nc)! * max(nv,nc)**3)`

            NOTE: This method returns exact value but if one of the nv, nc is greater than
            10 it is extremely slow.

            `"aa"` - implements Alternating Algorithm heuristic described in paper

            Vladyslav Sokol, Ante Ćustić, Abraham P. Punnen, Binay Bhattacharya (2020) Bilinear
            Assignment Problem: Large Neighborhoods and Experimental Analysis of Algorithms. INFORMS
            Journal on Computing 32(3):730-746. https://doi.org/10.1287/ijoc.2019.0893

            which solves the equivalent Bilinear Assignment Problem (BAP). The algorithm first
            generates a feasible solution to the BAP by sampling from a uniform distribution two
            permutations σ, v and then performs a coordinate-descent-like refinment by
            interchangeably fixing one of the permutations, solving the corresponding Linear
            Assignment Problem (LAP) and updating the other permutation with the matching found in
            LAP, doing so until convergence. Time complexity of this method is `O(N * (nv**3 +
            nc**3))` where N is the number of iterations it takes for the algorithm to converge.

            NOTE: This method is much faster in practice than "bf" but there are no theoretical
            guarantees on the approximation ratio for the used heuristic.

            `"bb"` - implements a branch-and-bound algorithm with problem specific branching rules
            and bounding function which solves the corresponding Bilinear Assignment Problem (BAP).
            For implementation details see 'bap.h' file. The "aa" heuristic is used in the "bb"
            method for getting initial upper bound.

            NOTE: This method returns exact value, however the performance may vary greatly
            depending on the specific problem instance. In general, due to the implemented bounding
            function, it is recommended that nv ~= nc. In asymmetrical case nv >> nc the "bf" method
            should provide better performance.

        repeats:
            Number of times we compute distance using "aa" method (i.e. we sample `repeats` starting
            permutations and then perform coordinate descent) and choose the smallest value.

            NOTE: If method not in ("aa", "bb") this option is ignored.

        seed:
            Seed of the PRNG used. Must be a non-negative integer or -1 for randomly set seed.

            NOTE: If method not in ("aa", "bb") this option is ignored.

    Returns:
        Isomorphic Spearman distance between U and V.

    Raises:
        ImportError: Raises exception if extension module is not found.
    """
    try:
        import fastmap._spear
    except ImportError as e:
        raise ImportError("Error while importing C extension for computing Isomorphic Spearman distance") from e

    assert isinstance(U, np.ndarray) and isinstance(V, np.ndarray), "Expected numpy arrays"
    assert U.shape == V.shape, "Expected arrays to have the same shape"
    assert (dim := len(U.shape)) == 2, f"Expected 2-D arrays, got {dim}-D arrays"
    assert type(seed) is int and seed >= -1, "Expected `seed` to be a non-negative integer or -1"
    assert type(repeats) is int and repeats > 0, "Expected `repeats` to be an int greater than 0"

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
        seed,
    )


def hamming(
    U: np.ndarray[int],
    V: np.ndarray[int],
    method: str = "bf",
    repeats: int = 30,
    seed: int = -1,
) -> int:
    """Computes Isomorphic Hamming distance between approval elections U and V.

    Computes Isomorphic Hamming distance between approval elections U and V defined as
    ```
    min_{v ∈ S_nv} min_{σ ∈ S_nc} sum_{i=0,..,nv-1} sum_{k=0,..,nc-1} d(i,v(i),k,σ(k))
    ```
    where d(i,j,k,l) := U[i,k] xor V[j,l], nc is the number of candidates, nv is the number of
    voters and S_n denotes the set of all permutations of the set {0,..,n-1}.

    NOTE: This function is a Python wrapper around C extension. For implementation details see
    'pyhamm.c' and 'bap.h' files.

    Args:
        U:
            Approval Election matrix s.t. U[i,j] ∈ {0,1} is equal to 1 if i-th approval ballot in
            the U election contains j-th candidate and 0 otherwise. Shape (nv, nc).

        V:
            Approval Election matrix s.t. V[i,j] ∈ {0,1} is equal to 1 if i-th approval ballot in
            the V election contains j-th candidate and 0 otherwise. Shape (nv, nc).

        method:
            Method used to compute the distance. Should be one of the

            `"bf"` - uses brute-force to solve the equivalent Bilinear Assignment Problem (BAP).
            Generates all permutations σ of the set {1,..,min(nv,nc)} using Heap's algorithm and for
            each generated permutation σ solves the Linear Assignment Problem (LAP) to obtain the
            optimal permutation v of {0,..,max(nv-1,nc-1)}. Time complexity of this method is
            `O(min(nv,nc)! * max(nv,nc)**3)`

            NOTE: This method returns exact value but if one of the nv, nc is greater than 10 it is
            extremely slow.

            `"aa"` - implements Alternating Algorithm heuristic described in paper

            Vladyslav Sokol, Ante Ćustić, Abraham P. Punnen, Binay Bhattacharya (2020) Bilinear
            Assignment Problem: Large Neighborhoods and Experimental Analysis of Algorithms. INFORMS
            Journal on Computing 32(3):730-746. https://doi.org/10.1287/ijoc.2019.0893

            which solves the equivalent Bilinear Assignment Problem (BAP). The algorithm first
            generates a feasible solution to the BAP by sampling from a uniform distribution two
            permutations σ, v and then performs a coordinate-descent-like refinment by
            interchangeably fixing one of the permutations, solving the corresponding Linear
            Assignment Problem (LAP) and updating the other permutation with the matching found in
            LAP, doing so until convergence. Time complexity of this method is O(N * (nv**3 +
            nc**3)) where N is the number of iterations it takes for the algorithm to converge.

            NOTE: This method is much faster in practice than "bf" but there are no theoretical
            guarantees on the approximation ratio for the used heuristic.

            `"bb"` - implements a branch-and-bound algorithm with problem specific branching rules
            and bounding function which solves the corresponding Bilinear Assignment Problem (BAP).
            For implementation details see 'bap.h' file. The "aa" heuristic is used in the "bb"
            method for getting initial upper bound.

            NOTE: This method returns exact value, however the performance may vary greatly
            depending on the specific problem instance. In general, due to the implemented bounding
            function, it is recommended to use this method if nv ≈ nc. In asymmetrical case nv >> nc
            the "bf" method should provide better performance.

        repeats:
            Number of times we compute distance using "aa" method (i.e. we sample `repeats` starting
            permutations and then perform coordinate descent) and choose the smallest value.

            NOTE: If method not in ("aa", "bb") this option is ignored.

        seed:
            Seed of the PRNG used. Must be a non-negative integer or -1 for randomly set seed.

            NOTE: If method not in ("aa", "bb") this option is ignored.

    Returns:
        Isomorphic Hamming distance between U and V.

    Raises:
        ImportError: Raises exception if extension module is not found.
    """
    try:
        import fastmap._hamm
    except ImportError as e:
        raise ImportError("Error while importing C extension for computing Isomorphic Hamming distance") from e

    assert isinstance(U, np.ndarray) and isinstance(V, np.ndarray), "Expected numpy arrays"
    assert U.shape == V.shape, "Expected arrays to have the same shape"
    assert (dim := len(U.shape)) == 2, f"Expected 2-D arrays, got {dim}-D arrays"
    assert type(seed) is int and seed >= -1, "Expected `seed` to be a non-negative integer or -1"
    assert type(repeats) is int and repeats > 0, "Expected `repeats` to be an int greater than 0"

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
        seed,
    )


def swap(
    U: np.ndarray[int],
    V: np.ndarray[int],
    method: str = "bf",
    repeats: int = 30,
    seed: int = -1,
) -> int:
    """Computes Isomorphic swap distance between ordinal elections U and V.

    Computes Isomorphic swap distance between ordinal elections U and V defined as
    ```
    min_{v ∈ S_nv} min_{σ ∈ S_nc} sum_{i=0,..,nv-1} sum_{k=0,..,nc-1} sum_{l=0,..,nc-1} d(i,v(i),k,l,σ(k),σ(l))
    ```
    where d(i,j,k,l,m,n) := 1/2 * { (pos_U[i,k] - pos_U[i,l]) * (pos_V[j,m] - pos_V[j,n]) < 0 }
    ({...} denoting here the Iverson bracket), nc is the number of candidates, nv is the number of
    voters, pos_U[i,k] denotes the position of k-th candidate in the i-th vote in the U election and
    S_n denotes the set of all permutations of the set {0,..,n-1}.

    NOTE: This function is a Python wrapper around C extension. For implementation details see
    'pyswap.c' file.

    Args:
        U:
            Ordinal Election matrix s.t. U[i,j] ∈ {0,..,nc-1} is the candidate's number on the j-th
            position in the i-th vote in the U election. Shape (nv, nc).

        V:
            Ordinal Election matrix s.t. V[i,j] ∈ {0,..,nc-1} is the candidate's number on the j-th
            position in the i-th vote in the V election. Shape (nv, nc).

        method:
            Method used to compute the distance. Should be one of the

            `"bf"` - uses a brute-force approach which generates all permutations of S_nc using
            Heap's algorithm and for every generated permutation σ solves the corresponding Linear
            Assignment Problem (LAP) in order to find the optimal permutation v. Depending on the
            number of candidates nc it either uses a dynamic approach to compute the cost matrix for
            LAP or uses a special, precomputed lookup table and dynamically computes the appropriate
            key to this lookup table to obtain cost matrix. The latter method is more efficient but
            has large memory requirements thus it is only used for up to nc=10. For details see
            'pyswap.c' file.

            `"aa"` - implements a coordinate-descent heuristic, analogous to the Alternating
            Algorithm for BAP problem (see docs for the fastmap.hamming or fastmap.spearman
            function). Time complexity of this method is `O(N * (nv**3 + nc**3 * nv + nv**2 *
            nc**2))` where N is the number of iterations it takes for the algorithm to converge.

            NOTE: This method is much faster in practice than "bf" but there are no theoretical
            guarantees on the approximation ratio for the used heuristic.

        repeats:
            Number of times we compute distance using "aa" method (i.e. we sample `repeats` starting
            permutations and then perform coordinate descent) and choose the smallest value.

            NOTE: If method not in ("aa",) this option is ignored.

        seed:
            Seed of the PRNG used. Must be a non-negative integer or -1 for randomly set seed.

            NOTE: If method not in ("aa",) this option is ignored.

    Returns:
        Isomorphic swap distance between U and V.

    Raises:
        ImportError: Raises exception if extension module is not found.
    """
    try:
        import fastmap._swap
    except ImportError as e:
        raise ImportError("Error while importing C extension for computing Isomorphic swap distance") from e

    assert isinstance(U, np.ndarray) and isinstance(V, np.ndarray), "Expected numpy arrays"
    assert U.shape == V.shape, "Expected arrays to have the same shape"
    assert (dim := len(U.shape)) == 2, f"Expected 2-D arrays, got {dim}-D arrays"
    assert type(seed) is int and seed >= -1, "Expected `seed` to be a non-negative integer or -1"
    assert type(repeats) is int and repeats > 0, "Expected `repeats` to be an int greater than 0"

    # We transpose the matrices so that the memory access pattern is better
    pos_U, pos_V = U.argsort().T, V.argsort().T

    return fastmap._swap.swap(
        pos_U.astype(np.int32),
        pos_V.astype(np.int32),
        {"bf": 0, "aa": 1}[method],
        # Options
        repeats,
        seed,
    )


def pairwise(
    M_U: np.ndarray[float],
    M_V: np.ndarray[float],
    method: str = "faq",
    repeats: int = 30,
    seed: int = -1,
    maxiter: int = 30,
    tol: float = 1e-3,
) -> float:
    """Computes pairwise L1 distance between ordinal elections U and V.

    Computes pairwise L1 distance between ordinal elections U and V defined as
    ```
    min_{σ ∈ S_nc} sum_{i=0,..,nc-1} sum_{j=0,..,nc-1} d(i,σ(i),j,σ(j))
    ```
    where d(i,j,k,l) := abs(M_U[i,k] - M_V[j,l]), nc is the number of candidates and S_n denotes the
    set of all permutations of the set {0,..,n-1}. Matrices M_U and M_V are the so called pairwise
    matrices of elections U and V. Pairwise matrix M of an election U is a matrix whose element
    M[i,j] is equal to the (normalized) number of votes in which i-th candidate comes before the
    j-th one.

    NOTE: This function is a Python wrapper around C extension. For implementation details see
    'qap.h' and 'pypairwise.c' files.

    Args:
        M_U:
            Normalized pairwise matrix of ordinal election U. Shape (nc, nc).

        M_V:
            Normalized pairwise matrix of ordinal election V. Shape (nc, nc).

        method:
            Method used to compute the distance. Should be one of the

            `"faq"` - implements generalized Fast Approximate QAP (FAQAP) algorithm for aproximately
            solving Lawler QAP (as opposed to more restricted Koopmans and Beckmann formulation)
            described in detail in paper

            Vogelstein JT, Conroy JM, Lyzinski V, Podrazik LJ, Kratzer SG, Harley ET, et al. (2015)
            Fast Approximate Quadratic Programming for Graph Matching. PLoS ONE 10(4): e0121002.
            https://doi.org/10.1371/journal.pone.0121002

            `"aa"` - implements a coordinate-descent heuristic, which solves a corresponding BAP
            problem using Alternating Algorithm (see docs for the fastmap.hamming or
            fastmap.spearman function). Time complexity of this method is `O(N * nc**3)` where N is
            the number of iterations it takes for the algorithm to converge.

        repeats:
            Number of times we compute distance using "aa" or "faq" method (i.e. we sample `repeats`
            starting permutations or doubly stochastic matrices and then perform refinement
            procedure) and choose the smallest value.

            NOTE: If method not in ("faq", "aa") this option is ignored.

        seed:
            Seed of the PRNG used. Must be a non-negative integer or -1 for randomly set seed.

            NOTE: If method not in ("faq", "aa") this option is ignored.

        maxiter:
            Integer specifying the max number of Frank-Wolfe iterations performed in the FAQ method
            of computing the pairwise distance.

            NOTE: If method not in ("faq",) this option is ignored.

        tol:
            Tolerance for termination in the FAQ method of computing the pairwise distance.
            Frank-Wolfe iteration terminates when ||P_{i} - P_{i+1}||_Frobenius < tol, where P is
            the solution to the relaxed QAP and i is the iteration number.

            NOTE: If method not in ("faq",) this option is ignored.

    Returns:
        Pairwise distance between two elections with pairwise matrices M_U and M_V.

    Raises:
        ImportError: Raises exception if extension module is not found.
    """
    try:
        import fastmap._pairwise
    except ImportError as e:
        raise ImportError("Error while importing C extension for computing pairwise distance") from e

    assert isinstance(M_U, np.ndarray) and isinstance(M_V, np.ndarray), "Expected numpy arrays"
    assert M_U.shape == M_V.shape, "Expected arrays to have the same shape"
    assert (dim := len(M_U.shape)) == 2, f"Expected 2-D arrays, got {dim}-D arrays"
    assert M_U.shape[0] == M_U.shape[1], "Expected pairwise matrix to be a square matrix"
    assert type(seed) is int and seed >= -1, "Expected `seed` to be a non-negative integer or -1"
    assert type(repeats) is int and repeats > 0, "Expected `repeats` to be an int greater than 0"
    assert type(maxiter) is int and maxiter > 0, "Expected `maxiter` to be an int greater than 0"
    assert type(tol) is float and tol > 0, "Expected `tol` to be a float greater than 0"

    return fastmap._pairwise.pairwise(
        M_U.astype(np.double),
        M_V.astype(np.double),
        {"faq": 0, "aa": 1}[method],
        # Options
        repeats,
        seed,
        maxiter,
        tol,
    )
