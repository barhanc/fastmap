#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "lap.h"

#define swap(type, x, y) \
    {                    \
        type SWAP = x;   \
        x = y;           \
        y = SWAP;        \
    }

/**
 * @brief TODO: Write a docstring
 *
 * @param pos_U
 * @param pos_V
 * @param nv
 * @param nc
 * @return int32_t
 */
static int32_t
swap_bf (const int32_t *pos_U, const int32_t *pos_V, const size_t nv, const size_t nc)
{
    // Cost matrix for LAP
    int32_t *cost = calloc (nv * nv, sizeof (int32_t));
    for (size_t k = 0; k < nc; k++)
        for (size_t l = k; l < nc; l++)
            for (size_t i = 0; i < nv; i++)
            {
                register int32_t r1 = pos_U[(i) + (k)*nv] - pos_U[(i) + (l)*nv];
                for (size_t j = 0; j < nv; j++)
                    cost[i * nv + j] += (r1 * (pos_V[(j) + (k)*nv] - pos_V[(j) + (l)*nv]) < 0);
            }

    // Stack pointer and encoding of the stack in iterative version of Heap's algorithm. See:
    // https://en.wikipedia.org/wiki/Heap%27s_algorithm
    size_t alpha = 1, *stack = calloc (nc, sizeof (size_t));

    // Permutation array initialized to identity permutation
    size_t *sigma = calloc (nc, sizeof (size_t));
    for (size_t i = 0; i < nc; i++)
        sigma[i] = i;

    // Indices of elements of permutation sigma which are swapped in the given iteration of Heap's
    // algorithm
    size_t p = 0, q = 0;

    // Auxiliary variables required for J-V LAP algorithm
    int32_t *a = calloc (nv, sizeof (int32_t));
    int32_t *b = calloc (nv, sizeof (int32_t));
    int32_t *x = calloc (nv, sizeof (int32_t));
    int32_t *y = calloc (nv, sizeof (int32_t));

    int32_t best_res = lap (nv, cost, a, b, x, y);

    while (alpha < nc)
    {
        if (stack[alpha] < alpha)
        {
            if (alpha % 2 == 0)
                p = 0, q = alpha;
            else
                p = alpha, q = stack[alpha];

            for (size_t k = 0; k < nc; k++)
                for (size_t i = 0; i < nv; i++)
                {
                    register int32_t r1 = pos_U[i + p * nv] - pos_U[i + k * nv];
                    register int32_t r2 = pos_U[i + q * nv] - pos_U[i + k * nv];

                    for (size_t j = 0; j < nv; j++)
                    {
                        int32_t r3 = pos_V[j + sigma[q] * nv] - pos_V[j + sigma[k] * nv];
                        int32_t r4 = pos_V[j + sigma[p] * nv] - pos_V[j + sigma[k] * nv];

                        cost[i * nv + j] += (r1 * r3 < 0) + (r2 * r4 < 0) - (r1 * r4 < 0) - (r2 * r3 < 0);
                    }
                }

            for (size_t i = 0; i < nv; i++)
            {
                register int32_t r1 = pos_U[i + q * nv] - pos_U[i + p * nv];

                for (size_t j = 0; j < nv; j++)
                {
                    int32_t r2 = pos_V[j + sigma[q] * nv] - pos_V[j + sigma[p] * nv];
                    cost[i * nv + j] += (r1 * r2 < 0) + (r1 * r2 > 0);
                }
            }

            swap (size_t, sigma[p], sigma[q]);

            int32_t res = lap (nv, cost, a, b, x, y);
            best_res = res < best_res ? res : best_res;

            stack[alpha]++;
            alpha = 1;
        }
        else
        {
            stack[alpha] = 0;
            alpha++;
        }
    }

    free (cost);
    free (stack);
    free (sigma);
    free (a);
    free (b);
    free (x);
    free (y);

    return best_res;
}

static const size_t pow10[] = {
    1,
    10,
    100,
    1000,
    10000,
    100000,
    1000000,
    10000000,
    100000000,
    1000000000,
};

/**
 * @brief Computes number of inversions (see:
 * https://en.wikipedia.org/wiki/Inversion_(discrete_mathematics)) for every permutation p of set
 * {0,..,n-1} and stores the result in `mem` array under index E(p) where E is an encoding of p
 * ```
 *  E(p) := 10**0 * p[0] + 10**1 * p[1] + ... + 10**(n-2) * p[n-2] .
 * ```
 * Notice here that specifying n-1 elements of p already uniquely defines p.
 * NOTE: we assume that n <= 10, otherwise the encoding E would not be unique. Notice also that we
 * could use a larger base (e.g. 11, 12) but for n=10 we already need to alocate 10**9B = 1GB
 * memory, thus using this method for larger permutations would require too much memory.
 *
 * @param n number of elements of permutation
 * @param mem pointer to an array used as a lookup table
 */
static void
mem_inversion_cnt (size_t n, uint8_t *mem)
{
    // Stack pointer and encoding of the stack in iterative version of Heap's algorithm. See:
    // https://en.wikipedia.org/wiki/Heap%27s_algorithm
    size_t alpha = 1, *stack = calloc (n, sizeof (size_t));

    // Permutation array initialized to identity permutation
    size_t *sigma = calloc (n, sizeof (size_t));
    for (size_t i = 0; i < n; i++)
        sigma[i] = i;

    // Indices of elements of permutation sigma which are swapped in the given iteration of Heap's
    // algorithm
    size_t p = 0, q = 0;

    while (alpha < n)
    {
        if (stack[alpha] < alpha)
        {
            if (alpha % 2 == 0)
                p = 0, q = alpha;
            else
                p = alpha, q = stack[alpha];

            swap (size_t, sigma[p], sigma[q]);

            uint8_t inversions = 0;
            for (size_t i = 0; i < n - 1; i++)
                for (size_t j = i + 1; j < n; j++)
                    inversions += sigma[i] > sigma[j] ? 1 : 0;

            size_t key = 0;
            for (size_t i = 0; i < n - 1; i++)
                key += sigma[i] * pow10[i];
            mem[key] = inversions;

            stack[alpha]++;
            alpha = 1;
        }
        else
        {
            stack[alpha] = 0;
            alpha++;
        }
    }

    free (stack);
    free (sigma);

    return;
}
/**
 * @brief TODO: Writer a docstring
 *
 * @param pos_U
 * @param nv
 * @param nc
 * @return int32_t
 */
static int32_t
swap_bf_mem (const int32_t *pos_U, const int32_t *pos_V, const size_t nv, const size_t nc)
{
    // Ordinal Election matrix of the 2nd election (V) constructed from position matrix pos_V
    int32_t *V = calloc (nv * nc, sizeof (size_t));
    for (size_t i = 0; i < nv; i++)
        for (size_t j = 0; j < nc; j++)
            V[i + nv * pos_V[i + nv * j]] = j;

    // A lookup table storing inversion counts for every permutation of set {0,..,nc-1}. The key to
    // the table is an encoded permutation.
    uint8_t *mem = calloc (pow10[nc - 1], sizeof (uint8_t));
    mem_inversion_cnt (nc, mem);

    // Cost matrix for LAP
    int32_t *cost = calloc (nv * nv, sizeof (int32_t));

    // Matrix of keys to the lookup table for every pair of votes. We utilize the fact that in every
    // iteration of Heap's algorihtm only two elements are swapped thus we can update key[i,j] in
    // O(1) and thus update cost[i,j] in O(1) instead of O(nc).
    size_t *key = calloc (nv * nv, sizeof (size_t));

    for (size_t i = 0; i < nv; i++)
        for (size_t j = 0; j < nv; j++)
        {
            for (size_t k = 0; k < nc - 1; k++)
                key[i * nv + j] += pos_U[i + nv * V[j + nv * k]] * pow10[k];
            cost[i * nv + j] = mem[key[i * nv + j]];
        }

    // Stack pointer and encoding of the stack in iterative version of Heap's algorithm. See:
    // https://en.wikipedia.org/wiki/Heap%27s_algorithm
    size_t alpha = 1, *stack = calloc (nc, sizeof (size_t));

    // Permutation array initialized to identity permutation
    size_t *sigma = calloc (nc, sizeof (size_t));
    for (size_t i = 0; i < nc; i++)
        sigma[i] = i;

    // Indices of elements of permutation sigma which are swapped in the given iteration of Heap's
    // algorithm
    size_t p = 0, q = 0;

    // Auxiliary variables required for J-V LAP algorithm
    int32_t *a = calloc (nv, sizeof (int32_t));
    int32_t *b = calloc (nv, sizeof (int32_t));
    int32_t *x = calloc (nv, sizeof (int32_t));
    int32_t *y = calloc (nv, sizeof (int32_t));

    int32_t best_res = lap (nv, cost, a, b, x, y);

    while (alpha < nc)
    {
        if (stack[alpha] < alpha)
        {
            if (alpha % 2 == 0)
                p = 0, q = alpha;
            else
                p = alpha, q = stack[alpha];

            for (size_t i = 0; i < nv; i++)
            {
                register int32_t r1 = pos_U[i + nv * q];
                register int32_t r2 = pos_U[i + nv * p];

                for (size_t j = 0; j < nv; j++)
                {
                    size_t pos_p = pos_V[j + nv * sigma[p]];
                    size_t pos_q = pos_V[j + nv * sigma[q]];

                    if (pos_p < nc - 1)
                        key[i * nv + j] += r1 * pow10[pos_p] - r2 * pow10[pos_p];

                    if (pos_q < nc - 1)
                        key[i * nv + j] += r2 * pow10[pos_q] - r1 * pow10[pos_q];

                    cost[i * nv + j] = mem[key[i * nv + j]];
                }
            }

            swap (size_t, sigma[p], sigma[q]);

            int32_t res = lap (nv, cost, a, b, x, y);
            best_res = res < best_res ? res : best_res;

            stack[alpha]++;
            alpha = 1;
        }
        else
        {
            stack[alpha] = 0;
            alpha++;
        }
    }

    free (V);
    free (key);
    free (mem);
    free (cost);
    free (stack);
    free (sigma);
    free (a);
    free (b);
    free (x);
    free (y);

    return best_res;
}

// =================================================================================================
// =================================================================================================

static PyObject *
py_swap (PyObject *self, PyObject *args)
{
    PyObject *result = NULL, *obj_X = NULL, *obj_Y = NULL;
    int method = 0, N_METHODS = 1;
    if (!PyArg_ParseTuple (args, "OOi", &obj_X, &obj_Y, &method))
        return NULL;

    PyArrayObject *obj_cont_X = (PyArrayObject *)PyArray_ContiguousFromAny (obj_X, NPY_INT32, 0, 0);
    PyArrayObject *obj_cont_Y = (PyArrayObject *)PyArray_ContiguousFromAny (obj_Y, NPY_INT32, 0, 0);
    if (!obj_cont_X || !obj_cont_Y)
        return NULL;

    if (method < 0 || method >= N_METHODS)
    {
        PyErr_Format (PyExc_ValueError, "expected method to be an int between 0 and %d", N_METHODS - 1);
        goto cleanup;
    }

    if (PyArray_NDIM (obj_cont_X) != 2 || PyArray_NDIM (obj_cont_Y) != 2)
    {
        PyErr_Format (PyExc_ValueError, "expected 2-D arrays, got a %d-D and %d-D array",
                      PyArray_NDIM (obj_cont_X), PyArray_NDIM (obj_cont_Y));
        goto cleanup;
    }

    int32_t *pos_U = (int32_t *)PyArray_DATA (obj_cont_X);
    int32_t *pos_V = (int32_t *)PyArray_DATA (obj_cont_Y);
    if (pos_U == NULL || pos_V == NULL)
    {
        PyErr_SetString (PyExc_TypeError, "invalid array object");
        goto cleanup;
    }

    npy_intp rows_X = PyArray_DIM (obj_cont_X, 0), cols_X = PyArray_DIM (obj_cont_X, 1);
    npy_intp rows_Y = PyArray_DIM (obj_cont_Y, 0), cols_Y = PyArray_DIM (obj_cont_Y, 1);
    if (rows_X != rows_Y || cols_X != cols_Y)
    {
        PyErr_SetString (PyExc_TypeError, "expected arrays to have the same shape");
        goto cleanup;
    }

    size_t nc = rows_X, nv = cols_X;
    int32_t ret = -1;

    Py_BEGIN_ALLOW_THREADS;
    switch (method)
    {
    case 0:
        if (nc <= 10) // See NOTE in the docstring of mem_inversion_cnt()
            ret = swap_bf_mem (pos_U, pos_V, nv, nc);
        else
            ret = swap_bf (pos_U, pos_V, nv, nc);
        break;
    default:
        break;
    }
    Py_END_ALLOW_THREADS;

    result = PyLong_FromLong (ret);

cleanup:
    Py_XDECREF ((PyObject *)obj_cont_X);
    Py_XDECREF ((PyObject *)obj_cont_Y);
    return result;
}

static PyMethodDef methods[] = {
    { "swap", (PyCFunction)py_swap, METH_VARARGS, "Swap distance.\n" },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_swap", "Swap distance.\n", -1, methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit__swap (void)
{
    import_array ();
    return PyModule_Create (&moduledef);
}