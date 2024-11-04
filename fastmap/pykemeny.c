
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// Here should be the implementation of local search for single k.
// First argument should be the linearized position matrix for the election.
// Also k-Kemeny distance for ordinal election is an integer!
static int32_t
kemenyk_ls (int32_t *pos_V, int k, ...)
{
}

// Here should be the implementation of local search for all k
// First argument should be the linearized position matrix for the election.
// One of the arguments shoud be the array in which we store the k-Kemeny values.
// Also k-Kemeny distance for ordinal election is an integer!
static void
kemeny_ls (int32_t *pos_V, int32_t *res, ...)
{
}

// Here should be the Python-C interface.
// Python interface should look sth like this
// ```
// fastmap.kemeny(
// V: np.ndarray[int],                      # Approval election matrix, but later to C function we pass pos_V = V.argsort(),
//                                          # basically V = election.votes but we don't want to explicitly import mapel
//                                          # (btw. it is already deprecated and the new package is called mapof)
//
// k: int | None = None,                    # If not None return value of k-Kemeny else return np.array with values of all k-Kemeny
//
// method: str = "ls",                      # Used method ls = local search
//
// starting: np.ndarray[int] | None = None, # Starting ranking
//
// ...options_for_chosen_method,            # Options for chosen method e.g. l in the local search implementation
//
// ) -> float | np.ndarray[float]
// )
// ```
//
static PyObject *
py_kemeny (PyObject *self, PyObject *args)
{
}