#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

double *X = NULL, *Y = NULL;
#define d(i, j, k, l) fabs (X[(i) * nc + (k)] - Y[(j) * nc + (l)])
#include "qap.h"

static PyObject *
py_pairwise (PyObject *self, PyObject *args)
{
    PyObject *result = NULL; // Return object of the Python function
    PyObject *obj_X = NULL;  // Argument object for the 1st matrix
    PyObject *obj_Y = NULL;  // Argument object for the 2nd matrix
    // Options
    int N_METHODS = 2;
    int method = 0;
    int seed = -1;
    int repeats = 0;
    size_t maxiter = 0;
    double tol = 0;

    // Parse arguments from Python to C

    if (!PyArg_ParseTuple (args, "OOiiiid", &obj_X, &obj_Y, &method, &repeats, &seed, &maxiter, &tol))
        return NULL;

    PyArrayObject *obj_cont_X = (PyArrayObject *)PyArray_ContiguousFromAny (obj_X, NPY_DOUBLE, 0, 0);
    PyArrayObject *obj_cont_Y = (PyArrayObject *)PyArray_ContiguousFromAny (obj_Y, NPY_DOUBLE, 0, 0);
    if (!obj_cont_X || !obj_cont_Y)
        return NULL;

    // Assert method is between 0 and N_METHODS-1

    if (method < 0 || method >= N_METHODS)
    {
        PyErr_Format (PyExc_ValueError, "expected method to be an int between 0 and %d", N_METHODS - 1);
        goto cleanup;
    }

    // Assert arrays are 2D

    if (PyArray_NDIM (obj_cont_X) != 2 || PyArray_NDIM (obj_cont_Y) != 2)
    {
        PyErr_Format (PyExc_ValueError, "expected 2-D arrays, got a %d-D and %d-D array",
                      PyArray_NDIM (obj_cont_X), PyArray_NDIM (obj_cont_Y));
        goto cleanup;
    }

    // Cast array objects to C arrays

    X = (double *)PyArray_DATA (obj_cont_X);
    Y = (double *)PyArray_DATA (obj_cont_Y);
    if (X == NULL || Y == NULL)
    {
        PyErr_SetString (PyExc_RuntimeError, "invalid array object");
        goto cleanup;
    }

    // Assert arrays have the same shape

    npy_intp rows_X = PyArray_DIM (obj_cont_X, 0), cols_X = PyArray_DIM (obj_cont_X, 1);
    npy_intp rows_Y = PyArray_DIM (obj_cont_Y, 0), cols_Y = PyArray_DIM (obj_cont_Y, 1);
    if (rows_X != rows_Y || cols_X != cols_Y)
    {
        PyErr_SetString (PyExc_ValueError, "expected arrays to have the same shape");
        goto cleanup;
    }

    // Assert arrays are square

    if (rows_X != cols_X)
    {
        PyErr_SetString (PyExc_ValueError, "expected square arrays");
        goto cleanup;
    }

    // Variables required for C functions

    size_t nc = rows_X;
    double ret = -1;

    // Set seed of the PRNG

    srand (seed > -1 ? seed : time (NULL));

    // Compute solution using C functions

    Py_BEGIN_ALLOW_THREADS;
    switch (method)
    {
    case 0:
        ret = qap_faq (nc, maxiter, tol);
        for (int i = 0; i < repeats - 1; i++)
        {
            double f = qap_faq (nc, maxiter, tol);
            ret = ret > f ? f : ret;
        }
        break;
    case 1:
        ret = qap_aa (nc);
        for (int i = 0; i < repeats - 1; i++)
        {
            double f = qap_aa (nc);
            ret = ret > f ? f : ret;
        }
        break;
    default:
        break;
    }
    Py_END_ALLOW_THREADS;

    result = PyFloat_FromDouble (ret);

cleanup:
    Py_XDECREF ((PyObject *)obj_cont_X);
    Py_XDECREF ((PyObject *)obj_cont_Y);
    return result;
}

static PyMethodDef methods[] = {
    { "pairwise", (PyCFunction)py_pairwise, METH_VARARGS, "Pairwise distance.\n" },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_pairwise", "Pairwise distance.\n", -1, methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit__pairwise (void)
{
    import_array ();
    return PyModule_Create (&moduledef);
}