#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

// #include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "lap/lap.h"

#define swap(type, x, y) \
    {                    \
        type SWAP = x;   \
        x = y;           \
        y = SWAP;        \
    }
#define Ten(i, j, k, l) Ten[i * (nv * nc * nc) + j * (nc * nc) + k * nc + l]

/**
 * @brief
 *
 * @param Ten 4-D tensor of distances where T(i,j,k,l) = d((vote=i, cand=k), (vote=j, cand=l))
 * @param nv Number of votes
 * @param nc Number of candidates
 * @return int d-Isomorphic distance with distance tensor given by Ten
 */
static int
bfcm (const int *Ten, const int nv, const int nc)
{
    // printf ("Available threads: %d\n", omp_get_max_threads ());
    int **cost = (int **)malloc (nv * sizeof (int *));
    for (int i = 0; i < nv; i++)
        cost[i] = (int *)malloc (nv * sizeof (int));

    for (int i = 0; i < nv; i++)
        for (int j = 0; j < nv; j++)
        {
            int acc = 0;
            for (int k = 0; k < nc; k++)
                acc += Ten (i, j, k, k);
            cost[i][j] = acc;
        }

    int stack[nc], sigma[nc], alpha = 1;
    memset (stack, 0, sizeof stack);
    for (int i = 0; i < nc; i++)
        sigma[i] = i;

    int a[nv], b[nv], _x[nv], _y[nv];
    int best_res = lap (nv, cost, a, b, _x, _y);

    while (alpha < nc)
    {
        if (stack[alpha] < alpha)
        {
            if (alpha % 2 == 0)
            {
                for (int i = 0; i < nv; i++)
                    for (int j = 0; j < nv; j++)
                    {
                        cost[i][j] += Ten (i, j, alpha, sigma[0]) + Ten (i, j, 0, sigma[alpha]);
                        cost[i][j] -= Ten (i, j, 0, sigma[0]) + Ten (i, j, alpha, sigma[alpha]);
                    }
                swap (int, sigma[0], sigma[alpha]);
            }
            else
            {
                for (int i = 0; i < nv; i++)
                    for (int j = 0; j < nv; j++)
                    {
                        cost[i][j] += Ten (i, j, alpha, sigma[stack[alpha]]) + Ten (i, j, stack[alpha], sigma[alpha]);
                        cost[i][j] -= Ten (i, j, alpha, sigma[alpha]) + Ten (i, j, stack[alpha], sigma[stack[alpha]]);
                    }
                swap (int, sigma[alpha], sigma[stack[alpha]]);
            }

            int res = lap (nv, cost, a, b, _x, _y);
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

    for (int i = 0; i < nv; i++)
        free (cost[i]);
    free (cost);

    return best_res;
}

// ========================================================
// ========================================================
// ========================================================
// ========================================================

static PyObject *
pybfcm (PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *result = NULL;
    PyObject *obj_Ten = NULL;
    static const char *kwlist[] = { (const char *)"dist_tensor", NULL };
    if (!PyArg_ParseTupleAndKeywords (args, kwargs, "O", (char **)kwlist, &obj_Ten))
        return NULL;

    PyArrayObject *obj_cont = (PyArrayObject *)PyArray_ContiguousFromAny (obj_Ten, NPY_INT, 0, 0);
    if (!obj_cont)
        return NULL;
    if (PyArray_NDIM (obj_cont) != 4)
    {
        PyErr_Format (PyExc_ValueError, "expected a 4-D tensor, got a %d array", PyArray_NDIM (obj_cont));
        goto cleanup;
    }

    int32_t *Ten = (int *)PyArray_DATA (obj_cont);
    if (Ten == NULL)
    {
        PyErr_SetString (PyExc_TypeError, "invalid distance tensor object");
        goto cleanup;
    }

    npy_intp num_ax0 = PyArray_DIM (obj_cont, 0);
    npy_intp num_ax1 = PyArray_DIM (obj_cont, 1);
    npy_intp num_ax2 = PyArray_DIM (obj_cont, 2);
    npy_intp num_ax3 = PyArray_DIM (obj_cont, 3);
    if (num_ax0 != num_ax1 || num_ax2 != num_ax3)
    {
        PyErr_SetString (PyExc_TypeError, "expected tensor of shape (nv,nv,nc,nc)");
        goto cleanup;
    }

    int ret;

    Py_BEGIN_ALLOW_THREADS

        ret
        = bfcm (Ten, num_ax0, num_ax2);

    Py_END_ALLOW_THREADS

        result
        = PyLong_FromLong (ret);

cleanup:
    Py_XDECREF ((PyObject *)obj_cont);
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
