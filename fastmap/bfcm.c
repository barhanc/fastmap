#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdlib.h>
#include <string.h>

#include "lap/lap.h"

#define swap(type, x, y) \
    {                    \
        type SWAP = x;   \
        x = y;           \
        y = SWAP;        \
    }
#define cost(i, j) cost[(i) * nv + (j)]
#define d(i, j, k, l) abs (X[(i) * nc + (k)] - Y[(j) * nc + (l)]) // Spearman distance
// #define d(i, j, k, l) X[(i) * nc + (k)] ^ Y[(j) * nc + (l)]       // Approval Hamming distance

/**
 * @brief Exhaustive search over all possible candidates' (votes') matchings with polynomial-time
 * procedure for finding optimal votes' (candidate's) matching to minimize the following objective
 *
 *  min_{v ∈ S_nv} min_{σ ∈ S_nc} sum_{i=1,..,nv} sum_{k=1,..,nc} d(i,v(i),k,σ(k))
 *
 * where d(i,j,k,l) is a distance tensor between elections e.g.
 *
 * -> Spearman distance: d(i,j,k,l) := |X(i,k) - Y(j,l)|
 *    X(i,k) := position of the k-th candidate in i-th vote in the 1st election
 *    Y(j,l) := position of the l-th candidate in j-th vote in the 2nd election
 *
 * -> Approval Hamming distance: d(i,j,k,l) = X(i,k) xor Y(j,l)
 *    X(i,k) := 1 if i-th approval ballot in the 1st election contains k-th candidate else 0
 *    Y(j,l) := 1 if j-th approval ballot in the 2nd election contains l-th candidate else 0
 *
 * Implements iterative Heap's algorithm for generating all possible permutations (not in
 * lexicographical order) and for every permutation (which is equivalent to some candidates' <<
 * votes'>> matching) solves the corresponding LAP (using JV algorithm) to find votes' (candidate's)
 * matching. The exhaustive search is performed over candidates or votes whichever size is lower.
 *
 * @param X Matrix representing 1st election required for computing distance tensor
 * @param Y Matrix representing 2nd election required for computing distance tensor
 * @param nv Number of votes (== rows(X) == rows(Y))
 * @param nc Number of candidates (== cols(X) == cols(Y))
 * @return int32_t
 */
static int32_t
bfcm (const int32_t *X, const int32_t *Y, size_t nv, size_t nc)
{
    if (nv < nc)
        swap (size_t, nc, nv);

    int32_t *cost = calloc (nv * nv, sizeof (int32_t));
    for (size_t i = 0; i < nv; i++)
        for (size_t j = 0; j < nv; j++)
            for (size_t k = 0; k < nc; k++)
                cost (i, j) += d (i, j, k, k);

    size_t alpha = 1;
    size_t *stack = calloc (nc, sizeof (size_t)), *sigma = calloc (nc, sizeof (size_t));
    for (size_t i = 0; i < nc; i++)
        sigma[i] = i;

    int32_t *a = calloc (nv, sizeof (int32_t)), *b = calloc (nv, sizeof (int32_t));
    int32_t *x = calloc (nv, sizeof (int32_t)), *y = calloc (nv, sizeof (int32_t));
    int32_t best_res = lap (nv, cost, a, b, x, y);

    while (alpha < nc)
    {
        if (stack[alpha] < alpha)
        {
            if (alpha % 2 == 0)
            {
                for (size_t i = 0; i < nv; i++)
                    for (size_t j = 0; j < nv; j++)
                    {
                        cost (i, j) += d (i, j, alpha, sigma[0]) + d (i, j, 0, sigma[alpha]);
                        cost (i, j) -= d (i, j, 0, sigma[0]) + d (i, j, alpha, sigma[alpha]);
                    }
                swap (size_t, sigma[0], sigma[alpha]);
            }
            else
            {
                for (size_t i = 0; i < nv; i++)
                    for (size_t j = 0; j < nv; j++)
                    {
                        cost (i, j) += d (i, j, alpha, sigma[stack[alpha]]) + d (i, j, stack[alpha], sigma[alpha]);
                        cost (i, j) -= d (i, j, alpha, sigma[alpha]) + d (i, j, stack[alpha], sigma[stack[alpha]]);
                    }
                swap (size_t, sigma[alpha], sigma[stack[alpha]]);
            }

            int32_t res = lap (nv, cost, a, b, x, y);
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

    free (cost), free (stack), free (sigma);
    free (a), free (b), free (x), free (y);
    return best_res;
}

// =================================================================================================
// =================================================================================================
// =================================================================================================

static PyObject *
pybfcm (PyObject *self, PyObject *args)
{
    PyObject *result = NULL;
    PyObject *obj_X = NULL;
    PyObject *obj_Y = NULL;
    if (!PyArg_ParseTuple (args, "OO", &obj_X, &obj_Y))
        return NULL;

    PyArrayObject *obj_cont_X = (PyArrayObject *)PyArray_ContiguousFromAny (obj_X, NPY_INT, 0, 0);
    PyArrayObject *obj_cont_Y = (PyArrayObject *)PyArray_ContiguousFromAny (obj_Y, NPY_INT, 0, 0);
    if (!obj_cont_X || !obj_cont_Y)
        return NULL;
    if (PyArray_NDIM (obj_cont_X) != 2 || PyArray_NDIM (obj_cont_Y) != 2)
    {
        PyErr_Format (PyExc_ValueError, "expected a 2-D arrays, got a %d and %d array",
                      PyArray_NDIM (obj_cont_X), PyArray_NDIM (obj_cont_Y));
        goto cleanup;
    }

    int32_t *X = (int32_t *)PyArray_DATA (obj_cont_X);
    int32_t *Y = (int32_t *)PyArray_DATA (obj_cont_Y);
    if (X == NULL || Y == NULL)
    {
        PyErr_SetString (PyExc_TypeError, "invalid matrix object");
        goto cleanup;
    }

    npy_intp rows_X = PyArray_DIM (obj_cont_X, 0), cols_X = PyArray_DIM (obj_cont_X, 1);
    npy_intp rows_Y = PyArray_DIM (obj_cont_Y, 0), cols_Y = PyArray_DIM (obj_cont_Y, 1);
    if (rows_X != rows_Y || cols_X != cols_Y)
    {
        PyErr_SetString (PyExc_TypeError, "expected arrays to have the same shape");
        goto cleanup;
    }

    size_t nv = rows_X, nc = cols_X;
    int ret;
    Py_BEGIN_ALLOW_THREADS
        ret
        = bfcm (X, Y, nv, nc);
    Py_END_ALLOW_THREADS

        result
        = PyLong_FromLong (ret);

cleanup:
    Py_XDECREF ((PyObject *)obj_cont_X);
    Py_XDECREF ((PyObject *)obj_cont_Y);
    return result;
}

static PyMethodDef bfcm_methods[] = {
    { "bfcm",
      (PyCFunction)pybfcm,
      METH_VARARGS,
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
