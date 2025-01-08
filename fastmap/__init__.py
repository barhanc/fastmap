"""
Fastmap provides optimized algorithms implementation for Maps of Elections framework.

Fastmap is a small Python library that provides optimized C implementations (as well as convenient
Python wrappers) of algorithms for computing structural similarity between elections used in the Map
of Elections framework, such as the isomorphic swap and Spearman distances, Hamming distance,
pairwise distance and k-Kemeny distance.
"""

import ctypes
from typing import Literal
import numpy as np
from mapel.elections.objects.Election import Election


def swap_distance_between_potes(pote_1: list[int], pote_2: list[int], method: Literal["bf", "ms", "pd"] = "bf") -> int:
    """Computes the exact or approximated swap distance between two lists of positional votes (potes).

    This function calculates an approximation of the swap distance between two lists of positional votes
    (potes). The swap distance is computed using different heuristics, which depend on the chosen `method`.

    NOTE: This function is a Python wrapper around a C extension. For implementation details, see the
    'fastmap._swap_distance_between_potes' module.

    Args:
        pote_1:
            A numpy array representing the first positional vote (pote_1). The array elements should
            represent the rank positions of candidates in the votes. Shape (n,).

        pote_2:
            A numpy array representing the second positional vote (pote_2), with the same shape as
            `pote_1`. Each element should represent the rank position of the corresponding candidate in the vote.

        method:
            The method used to calculate the swap distance. Should be one of the following:

            `"bf"` - Brute-force approach that calculates the swap distance by checking all possible swaps
            between elements of `pote_1` and `pote_2`. This method works well for small lists but is computationally expensive for larger ones.

            `"pd"` - Positional difference method. This method approximates the swap distance by summing the
            absolute differences in corresponding positions in `pote_1` and `pote_2`. It provides a fast
            approximation for the swap distance but may not capture all the structural differences.

            `"ms"` - Merge-sort based approach that counts the number of inversions between `pote_1` and `pote_2`.
            It is particularly efficient for larger lists but may not be the most precise for small lists.

    Returns:
        int:
            The calculated exact or approximated swap distance between `pote_1` and `pote_2` based on the selected `method`.

    Raises:
        ImportError:
            If the C extension for computing the swap distance cannot be imported or is not available.

        AssertionError:
            If the input arrays do not meet the expected conditions (e.g., they are not numpy arrays, they have
            different shapes, or the `method` is not one of the valid options).
    """
    try:
        import fastmap._swap_distance_between_potes
    except ImportError as e:
        raise ImportError("Error while importing C extension for computing Approx Swap Distance between Potes") from e
    
    methods: dict[str, int] = {
        "bf": 0,
        "pd": 1,
        "ms": 2
    }

    # assert isinstance(pote_1, np.ndarray) and isinstance(pote_2, np.ndarray), "Expected numpy arrays"
    # assert pote_1.shape == pote_2.shape, "Expected arrays to have the same shape"
    assert method in methods, "Expected one of available methods: 'bf', 'pd', 'ms'"
    return fastmap._swap_distance_between_potes.swap_distance_between_potes(pote_1, pote_2, methods[method])
    

def local_search_kKemeny_single_k(election: Election, k: int, l: int, starting: np.ndarray[int] | None = None) -> dict[str, int]:
    """Performs local search for k-Kemeny problem optimization using a single k.

    Args:
        election:
            Election object from mapof library.
        k:
            The number of candidates to consider in the local search.
        l:
            A parameter controlling the local search's stopping criteria.
        starting:
            An optional 1-D numpy array of integers indicating the starting ranking (default is None).

    Returns:
        An integer representing the minimum distance between voters in the k-Kemeny problem optimization.

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
        starting_ptr = list(range(election.num_candidates))
    result = fastmap._local_search_kkemeny_single_k.local_search_kkemeny_single_k(
        election.votes, k, l, election.num_voters, election.num_candidates, starting_ptr
    )
    
    return {'value': result}


def simulated_annealing(
    election: Election, 
    initial_temp: float = 50.0, 
    cooling_rate: float = 0.995, 
    max_iterations: int = 500
) -> dict[str, list[int]]:
    """
    Performs simulated annealing for Kemeny ranking optimization.

    Args:
        rankings:
            A 2D list of integers representing rankings for the election.
        n:
            The number of items (candidates) in each ranking.
        initial_temp:
            The initial temperature for the simulated annealing process.
        cooling_rate:
            The rate at which the temperature is reduced during annealing.
        max_iterations:
            The maximum number of iterations for the annealing process.

    Returns:
        An integer representing the minimum distance between voters in the Kemeny problem optimization.

    Raises:
        ImportError: If the C extension module for simulated annealing is not found.
        ValueError: If input rankings are invalid or improperly formatted.

    Notes:
        - This function wraps a C extension to optimize performance.
        - Rankings should be a complete and valid 2D list of integers.
    """
    try:
        import simulated_annealing
    except ImportError as e:
        raise ImportError("C extension for simulated annealing could not be imported.") from e
    
    assert isinstance(election, Election), "Expected an Election object"

    # Call the C extension function
    try:
        best_ranking = simulated_annealing.simulated_annealing(
            election.votes, election.num_candidates, initial_temp, cooling_rate, max_iterations
        )
    except Exception as e:
        raise ValueError("Error during simulated annealing process.") from e

    # Calculate Kemeny distance for the best ranking
    return {"value": best_ranking}



def spearman(
    U: np.ndarray[int],
    V: np.ndarray[int],
    method: Literal["bf", "aa", "bb"] = "bf",
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
    method: Literal["bf", "aa", "bb"] = "bf",
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
    method: Literal["bf", "aa"] = "bf",
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
    method: Literal["faq", "aa"] = "faq",
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
