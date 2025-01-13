#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * Function: merge
 * ----------------
 * Merges two sorted subarrays of `pote_1` and `pote_2` while counting the inversions between them.
 * Inversions occur when the order between elements in `pote_1` and `pote_2` is inconsistent.
 *
 * Parameters:
 *  - `pote_1`: Pointer to the first positional vote.
 *  - `pote_2`: Pointer to the second positional vote.
 *  - `left`: Starting index of the subarray.
 *  - `mid`: Midpoint index dividing the subarrays.
 *  - `right`: Ending index of the subarray.
 *
 * Returns:
 *  - int: The number of inversions found during the merge.
 *
 * Complexity:
 *  - Time: O(n), where `n` is the size of the subarray.
 *  - Space: O(n), due to the use of temporary arrays.
 *
 * Notes:
 *  - This function is internally used by `merge_and_count` for recursive inversion counting.
 */
static inline int merge(int *pote_1, int *pote_2, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Temporary arrays
    int *temp_1 = (int *)malloc(n1 * sizeof(int));
    int *temp_2 = (int *)malloc(n1 * sizeof(int));
    int *temp_3 = (int *)malloc(n2 * sizeof(int));
    int *temp_4 = (int *)malloc(n2 * sizeof(int));

    // Copy data to temporary arrays
    for (int i = 0; i < n1; i++) {
        temp_1[i] = pote_1[left + i];
        temp_2[i] = pote_2[left + i];
    }
    for (int j = 0; j < n2; j++) {
        temp_3[j] = pote_1[mid + 1 + j];
        temp_4[j] = pote_2[mid + 1 + j];
    }

    int i = 0, j = 0, k = left, inv_count = 0;

    // Merge temporary arrays back into pote_1 and pote_2
    while (i < n1 && j < n2) {
        if (temp_1[i] <= temp_3[j]) {
            pote_1[k] = temp_1[i];
            pote_2[k] = temp_2[i];
            i++;
        } else {
            pote_1[k] = temp_3[j];
            pote_2[k] = temp_4[j];
            j++;
            inv_count += (n1 - i);  // Count inversions
        }
        k++;
    }

    // Copy the remaining elements
    while (i < n1) {
        pote_1[k] = temp_1[i];
        pote_2[k] = temp_2[i];
        i++;
        k++;
    }
    while (j < n2) {
        pote_1[k] = temp_3[j];
        pote_2[k] = temp_4[j];
        j++;
        k++;
    }

    // Free temporary arrays
    free(temp_1);
    free(temp_2);
    free(temp_3);
    free(temp_4);

    return inv_count;
}

/*
 * Function: merge_and_count
 * --------------------------
 * Recursively divides the array into subarrays, counts inversions within subarrays, and merges them
 * using the `merge` function to count cross-inversions.
 *
 * Parameters:
 *  - `pote_1`: Pointer to the first positional vote.
 *  - `pote_2`: Pointer to the second positional vote.
 *  - `left`: Starting index of the subarray.
 *  - `right`: Ending index of the subarray.
 *
 * Returns:
 *  - int: The total number of inversions in the given range `[left, right]`.
 *
 * Complexity:
 *  - Time: O(n log n), where `n` is the size of the array.
 *  - Space: O(n), due to temporary storage during merging.
 *
 * Notes:
 *  - This function is part of the merge-sort-based inversion counting method.
 */
static inline int merge_and_count(int *pote_1, int *pote_2, int left, int right) {
    if (left >= right) {
        return 0;
    }

    int mid = (left + right) / 2;
    int inv_count = 0;

    inv_count += merge_and_count(pote_1, pote_2, left, mid);
    inv_count += merge_and_count(pote_1, pote_2, mid + 1, right);
    inv_count += merge(pote_1, pote_2, left, mid, right);

    return inv_count;
}

/*
 * Function: swap_distance_between_potes_ms
 * ----------------------------------------
 * Heurestic approximating the swap distance (number of inversions) between two lists of positional votes using
 * a merge-sort-based approach.
 *
 * Parameters:
 *  - `pote_1`: Pointer to the first positional vote.
 *  - `pote_2`: Pointer to the second positional vote.
 *  - `n`: The length of the arrays.
 *
 * Returns:
 *  - int: The number of inversions between `pote_1` and `pote_2`.
 *
 * Complexity:
 *  - Time: O(n log n), where `n` is the size of the input arrays.
 *  - Space: O(n), due to temporary storage for copying the arrays.
 *
 * Notes:
 *  - This method approximates swap distance between two potes by calculating the number of inversions between them. It
 * works really well for pote lengths of >100 where approx ratio starts converging towards 1, but becomes inefficient
 * and inaccurate for smaller lengths, where Positional Distance Heuristic or exact method will perform better.
 */
static inline int swap_distance_between_potes_ms(int *pote_1, int *pote_2, int n) {
    int *pote_1_copy = (int *)malloc(n * sizeof(int));
    int *pote_2_copy = (int *)malloc(n * sizeof(int));

    // Copy arrays
    memcpy(pote_1_copy, pote_1, n * sizeof(int));
    memcpy(pote_2_copy, pote_2, n * sizeof(int));

    // Call merge_and_count
    int distance = merge_and_count(pote_1_copy, pote_2_copy, 0, n - 1);

    // Free memory
    free(pote_1_copy);
    free(pote_2_copy);

    return distance;
}

/*
 * Function: swap_distance_between_potes_pd
 * ----------------------------------------
 * Computes a heuristic estimate of the swap distance between two lists of positional votes by
 * summing the absolute positional differences.
 *
 * Parameters:
 *  - `potes1`: Pointer to the first positional vote.
 *  - `potes2`: Pointer to the second positional vote.
 *  - `n`: The length of the arrays.
 *
 * Returns:
 *  - int: The heuristic estimate of the swap distance.
 *
 * Complexity:
 *  - Time: O(n), where `n` is the size of the input arrays.
 *  - Space: O(1), no additional memory allocation.
 *
 * Notes:
 *  - This heuristic is really fast and work pretty well for small length values, but growing size the approx ratio
 * doesn't converge and stays somewhere in the range of 1.25-1.50.
 */
int swap_distance_between_potes_pd(int *potes1, int *potes2, int n) {
    int heuristic_value = 0;

    for (int i = 0; i < n; i++) {
        heuristic_value += abs(potes1[i] - potes2[i]);
    }

    return heuristic_value;
}

/*
 * Function: swap_distance_between_potes_bf
 * ----------------------------------------
 * Computes the swap distance between two lists of positional votes using a brute-force approach.
 *
 * Parameters:
 *  - `pote1`: Pointer to the first array vote.
 *  - `pote2`: Pointer to the second positional vote.
 *  - `n`: The length of the arrays.
 *
 * Returns:
 *  - int: The swap distance.
 *
 * Complexity:
 *  - Time: Triangular O(n^2), where `n` is the size of the input arrays.
 *  - Space: O(1), no additional memory allocation.
 *
 * Notes:
 *  - This method gives the exact swap distance between potes.
 */
static inline int swap_distance_between_potes_bf(int *pote1, int *pote2, int n) {
    int swap_distance = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if ((pote1[i] > pote1[j] && pote2[i] < pote2[j]) || (pote1[i] < pote1[j] && pote2[i] > pote2[j])) {
                swap_distance++;
            }
        }
    }

    return swap_distance;
}

/*
 * Function: py_swap_distance_between_potes
 * ----------------------------------------
 * Python wrapper for calculating swap distance or heuristics between two positional votes (potes).
 * Supports three methods: brute force, positional difference heuristic, and merge-sort inversion counting.
 *
 * Parameters:
 *  - `args`: Python tuple containing:
 *      1. `potes1` (list[int]): First positional vote.
 *      2. `potes2` (list[int]): Second positional vote.
 *      3. `method` (int): The computation method to use:
 *          - 0: Brute-force swap distance calculation.
 *          - 1: Positional difference heuristic.
 *          - 2: Merge-sort inversion counting heuristic.
 *
 * Returns:
 *  - Python integer: The computed swap distance or heuristic value.
 *
 * Raises:
 *  - `ValueError`: If input arguments are invalid (e.g., non-lists or mismatched lengths).
 *  - `MemoryError`: If memory allocation fails.
 *
 * Complexity:
 *  - Depends on the selected method:
 *      - Brute Force: O(n^2)
 *      - Positional Difference: O(n)
 *      - Merge Sort: O(n log n)
 *
 * Example Usage in Python:
 * ```python
 * import swap_distance
 *
 * potes1 = [3, 1, 2, 4]
 * potes2 = [1, 2, 3, 4]
 *
 * ### Brute force
 * print(_swap_distance_between_potes.py_swap_distance_between_potes(potes1, potes2, 0))
 *
 * ### Positional difference heuristic
 * print(_swap_distance_between_potes.py_swap_distance_between_potes(potes1, potes2, 1))
 *
 * ### Merge-sort inversion counting heuristic
 * print(_swap_distance_between_potes.py_swap_distance_between_potes(potes1, potes2, 2))
 * ```
 */
static PyObject *py_swap_distance_between_potes(PyObject *self, PyObject *args) {
    PyObject *result = NULL;
    PyObject *obj_potes1 = NULL;
    PyObject *obj_potes2 = NULL;
    int method = 0;
    size_t length = 0;

    // Parse Python arguments
    if (!PyArg_ParseTuple(args, "OOi", &obj_potes1, &obj_potes2, &method)) {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments. Expected (list, list, method).");
        return NULL;
    }

    // Convert Python lists to C arrays
    if (!PyList_Check(obj_potes1) || !PyList_Check(obj_potes2)) {
        PyErr_SetString(PyExc_TypeError, "Arguments must be lists.");
        return NULL;
    }

    length = PyList_Size(obj_potes1);
    if (length != (size_t)PyList_Size(obj_potes2)) {
        PyErr_SetString(PyExc_ValueError, "Lists must have the same length.");
        return NULL;
    }

    int *potes1 = (int *)malloc(length * sizeof(int));
    int *potes2 = (int *)malloc(length * sizeof(int));
    if (!potes1 || !potes2) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory.");
        free(potes1);
        free(potes2);
        return NULL;
    }

    // Populate C arrays from Python lists
    for (size_t i = 0; i < length; i++) {
        potes1[i] = (int)PyLong_AsLong(PyList_GetItem(obj_potes1, i));
        potes2[i] = (int)PyLong_AsLong(PyList_GetItem(obj_potes2, i));
    }

    // Call the appropriate method
    int result_value = -1;
    switch (method) {
        case 0:
            result_value = swap_distance_between_potes_bf(potes1, potes2, length);
            break;
        case 1:
            result_value = swap_distance_between_potes_pd(potes1, potes2, length);
            break;
        case 2:
            result_value = swap_distance_between_potes_ms(potes1, potes2, length);
            break;
        default:
            PyErr_SetString(PyExc_ValueError, "Invalid method. Supported methods: 0, 1, 2.");
            free(potes1);
            free(potes2);
            return NULL;
    }

    // Free allocated memory
    free(potes1);
    free(potes2);

    // Return result as a Python integer
    result = PyLong_FromLong(result_value);
    return result;
}

// Method definition table
static PyMethodDef methods[] = {
    {"swap_distance_between_potes", (PyCFunction)py_swap_distance_between_potes, METH_VARARGS,
     "Calculate exact or approximated swap distance between two potes."},
    {NULL, NULL, 0, NULL}  // Sentinel
};

// Module definition
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_swap_distance_between_potes",                                  // Module name
    "Calculate exact or approximated swap distance between potes.",  // Module docstring
    -1,  // Size of per-interpreter state of the module, or -1 if module keeps state in global variables
    methods};

// Module initialization function
PyMODINIT_FUNC PyInit__swap_distance_between_potes(void) {
    import_array();  // Initialize the NumPy C API
    return PyModule_Create(&moduledef);
}
