#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

int32_t *X = NULL, *Y = NULL;
#define d(i, j, k, l) abs (X[(i) * nc + (k)] - Y[(j) * nc + (l)])
#include "bap.h"

static PyObject *
py_spear (PyObject *self, PyObject *args)
{
    PyObject *result = NULL, *obj_X = NULL, *obj_Y = NULL;
    int method = 0, N_METHODS = 2;
    if (!PyArg_ParseTuple (args, "OOi", &obj_X, &obj_Y, &method))
        return NULL;

    if (method < 0 || method >= N_METHODS)
    {
        PyErr_Format (PyExc_ValueError, "expected method to be between 0 and %d", N_METHODS - 1);
        return NULL;
    }

    PyArrayObject *obj_cont_X = (PyArrayObject *)PyArray_ContiguousFromAny (obj_X, NPY_INT32, 0, 0);
    PyArrayObject *obj_cont_Y = (PyArrayObject *)PyArray_ContiguousFromAny (obj_Y, NPY_INT32, 0, 0);
    if (!obj_cont_X || !obj_cont_Y)
        return NULL;

    if (PyArray_NDIM (obj_cont_X) != 2 || PyArray_NDIM (obj_cont_Y) != 2)
    {
        PyErr_Format (PyExc_ValueError, "expected 2D arrays, got a %d and %d array",
                      PyArray_NDIM (obj_cont_X), PyArray_NDIM (obj_cont_Y));
        goto cleanup;
    }

    X = (int32_t *)PyArray_DATA (obj_cont_X);
    Y = (int32_t *)PyArray_DATA (obj_cont_Y);
    if (X == NULL || Y == NULL)
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

    size_t nv = rows_X, nc = cols_X;
    int32_t ret = -1;

    Py_BEGIN_ALLOW_THREADS;
    switch (method)
    {
    case 0:
        ret = bap_bf (nv, nc);
        break;
    case 1:
        ret = bap_aa (nv, nc);
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

static PyMethodDef fast_methods[] = {
    { "spear", (PyCFunction)py_spear, METH_VARARGS, "Spearman distance.\n" },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_spear", "Spearman distance.\n", -1, fast_methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit__spear (void)
{
    import_array ();
    return PyModule_Create (&moduledef);
}
