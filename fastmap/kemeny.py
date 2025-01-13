import ctypes
import numpy as np

from typing import Literal, Optional
from mapel.elections.objects.Election import Election


def swap_distance_between_potes(
    pote_1: list[int],
    pote_2: list[int],
    method: Literal["bf", "ms", "pd"] = "bf",
) -> int:
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

    methods: dict[str, int] = {"bf": 0, "pd": 1, "ms": 2}

    # assert isinstance(pote_1, np.ndarray) and isinstance(pote_2, np.ndarray), "Expected numpy arrays"
    # assert pote_1.shape == pote_2.shape, "Expected arrays to have the same shape"
    assert method in methods, "Expected one of available methods: 'bf', 'pd', 'ms'"
    return fastmap._swap_distance_between_potes.swap_distance_between_potes(pote_1, pote_2, methods[method])


def local_search_kKemeny_single_k(
    election: Election,
    k: int,
    l: int,
    starting: Optional[np.ndarray[int]] = None,
) -> dict[str, int]:
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

    return {"value": result}


def simulated_annealing(
    election: Election,
    initial_temp: float = 50.0,
    cooling_rate: float = 0.995,
    max_iterations: int = 500,
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
        import fastmap._simulated_annealing
    except ImportError as e:
        raise ImportError("C extension for simulated annealing could not be imported.") from e

    assert isinstance(election, Election), "Expected an Election object"

    # Call the C extension function
    try:
        best_ranking = fastmap._simulated_annealing.simulated_annealing(
            election.votes, election.num_candidates, initial_temp, cooling_rate, max_iterations
        )
    except Exception as e:
        raise ValueError("Error during simulated annealing process.") from e

    # Calculate Kemeny distance for the best ranking
    return {"value": best_ranking}
