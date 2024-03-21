#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

// #include <omp.h> // Not on Mac OS (at least not out-of-the-box)
#include <stdlib.h>
#include <string.h>

#include "lap/lap.h"

#define swap(type, x, y) \
    {                    \
        type SWAP = x;   \
        x = y;           \
        y = SWAP;        \
    }
#define d(i, j, k, l) abs (pos_x[i * nc + k] - pos_y[j * nc + l]) // Spearman distaance
// #define d(i, j, k, l) (pos_x[i * nc + k] == pos_y[j * nc + l] ? 1 : 0) // Hamming distance

/**
 * @brief Exhaustive search over all possible candidates' matchings with polynomial-time finding of
 * optimal votes' matching to minimize the following objective
 *
 *  Sum_{i,j=1,..,nv} Sum_{k,l=1,..,nc} d(i,j,k,l)
 *
 * where d(i,j,k,l) is a distance tensor between elections e.g.
 *
 *  Spearman distance: d(i,j,k,l) := |pos_x(i,k) - pos_y(j,l)|
 *  Hamming distance : d(i,j,k,l) := [pos_x(i,k) == pos_y(j,l)]
 *
 * Implements iterative Heap's algorithm for generating all possible permutations (not in
 * lexicographical order) and for every permutation (which is equivalent to some candidates'
 * matching) solves the corresponding LAP (using JV algorithm) to find votes' matching.
 *
 * @param pos_x Matrix of positions of each candidate in each vote in the 1st election
 * @param pos_y Matrix of positions of each candidate in each vote in the 2nd election
 * @param nv Number of votes (== rows(pos_x) == rows(pos_y))
 * @param nc Number of candidates (== cols(pos_x) == cols(pos_y))
 * @return int32_t
 */
static int32_t
bfcm (const int32_t *pos_x, const int32_t *pos_y, const size_t nv, const size_t nc)
{
    // printf ("Available threads: %d\n", omp_get_max_threads ());
    int32_t **cost = (int32_t **)malloc (nv * sizeof (int32_t *));
    for (size_t i = 0; i < nv; i++)
        cost[i] = (int32_t *)malloc (nv * sizeof (int32_t));

    for (size_t i = 0; i < nv; i++)
        for (size_t j = 0; j < nv; j++)
        {
            int32_t acc = 0;
            for (size_t k = 0; k < nc; k++)
                acc += d (i, j, k, k);
            cost[i][j] = acc;
        }

    int32_t stack[nc], sigma[nc], alpha = 1;
    memset (stack, 0, sizeof stack);
    for (size_t i = 0; i < nc; i++)
        sigma[i] = i;

    int32_t a[nv], b[nv], _x[nv], _y[nv];
    int32_t best_res = lap (nv, cost, a, b, _x, _y);

    while (alpha < nc)
    {
        if (stack[alpha] < alpha)
        {
            if (alpha % 2 == 0)
            {
                for (size_t i = 0; i < nv; i++)
                    for (size_t j = 0; j < nv; j++)
                    {
                        cost[i][j] += d (i, j, alpha, sigma[0]) + d (i, j, 0, sigma[alpha]);
                        cost[i][j] -= d (i, j, 0, sigma[0]) + d (i, j, alpha, sigma[alpha]);
                    }
                swap (int32_t, sigma[0], sigma[alpha]);
            }
            else
            {
                for (size_t i = 0; i < nv; i++)
                    for (size_t j = 0; j < nv; j++)
                    {
                        cost[i][j] += d (i, j, alpha, sigma[stack[alpha]]) + d (i, j, stack[alpha], sigma[alpha]);
                        cost[i][j] -= d (i, j, alpha, sigma[alpha]) + d (i, j, stack[alpha], sigma[stack[alpha]]);
                    }
                swap (int32_t, sigma[alpha], sigma[stack[alpha]]);
            }

            int32_t res = lap (nv, cost, a, b, _x, _y);
            best_res = (res < best_res ? res : best_res);
            stack[alpha]++;
            alpha = 1;
        }
        else
        {
            stack[alpha] = 0;
            alpha++;
        }
    }

    for (size_t i = 0; i < nv; i++)
        free (cost[i]);
    free (cost);

    return best_res;
}

// ========================================================
// ========================================================
// ========================================================
// ========================================================

static PyObject *
pybfcm (PyObject *self, PyObject *args)
{
    PyObject *result = NULL;
    PyObject *obj_pos_x = NULL;
    PyObject *obj_pos_y = NULL;
    if (!PyArg_ParseTuple (args, "OO", &obj_pos_x, &obj_pos_y))
        return NULL;

    PyArrayObject *obj_cont_x = (PyArrayObject *)PyArray_ContiguousFromAny (obj_pos_x, NPY_INT, 0, 0);
    PyArrayObject *obj_cont_y = (PyArrayObject *)PyArray_ContiguousFromAny (obj_pos_y, NPY_INT, 0, 0);
    if (!obj_cont_x || !obj_cont_y)
        return NULL;
    if (PyArray_NDIM (obj_cont_x) != 2 || PyArray_NDIM (obj_cont_y) != 2)
    {
        PyErr_Format (PyExc_ValueError, "expected a 2-D arrays, got a %d and %d array",
                      PyArray_NDIM (obj_cont_x), PyArray_NDIM (obj_cont_y));
        goto cleanup;
    }

    int32_t *pos_x = (int32_t *)PyArray_DATA (obj_cont_x);
    int32_t *pos_y = (int32_t *)PyArray_DATA (obj_cont_y);
    if (pos_x == NULL || pos_y == NULL)
    {
        PyErr_SetString (PyExc_TypeError, "invalid distance tensor object");
        goto cleanup;
    }

    npy_intp rows_x = PyArray_DIM (obj_cont_x, 0), cols_x = PyArray_DIM (obj_cont_x, 1);
    npy_intp rows_y = PyArray_DIM (obj_cont_y, 0), cols_y = PyArray_DIM (obj_cont_y, 1);

    if (rows_x != rows_y || cols_x != cols_y)
    {
        PyErr_SetString (PyExc_TypeError, "expected arrays to be oh the same shape");
        goto cleanup;
    }

    size_t nv = rows_x, nc = cols_x;
    int ret;
    Py_BEGIN_ALLOW_THREADS
        ret
        = bfcm (pos_x, pos_y, nv, nc);
    Py_END_ALLOW_THREADS

        result
        = PyLong_FromLong (ret);

cleanup:
    Py_XDECREF ((PyObject *)obj_cont_x);
    Py_XDECREF ((PyObject *)obj_cont_y);
    return result;
}

static PyMethodDef bfcm_methods[] = {
    { "bfcm",
      (PyCFunction)pybfcm,
      METH_VARARGS | METH_KEYWORDS,
      "Exhaustive seach over all candidates matchings.\n" },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "fastmap",
    "Exhaustive seach over all candidates matchings",
    -1,
    bfcm_methods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC
PyInit_bfcm (void)
{
    import_array ();
    return PyModule_Create (&moduledef);
}
