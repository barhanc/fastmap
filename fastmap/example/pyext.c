#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// =================================================================================================
// Include headers defining functions computing various distances
// =================================================================================================
#include "spear.h"

// =================================================================================================
// For every function define a Python wrapper for this function
// =================================================================================================

static PyObject *
py_spear (PyObject *self, PyObject *args)
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
        = spear (X, Y, nv, nc);
    Py_END_ALLOW_THREADS

        result
        = PyLong_FromLong (ret);

cleanup:
    Py_XDECREF ((PyObject *)obj_cont_X);
    Py_XDECREF ((PyObject *)obj_cont_Y);
    return result;
}

// =================================================================================================
// Define Python extension module with function (methods) computing various distances
// =================================================================================================

static PyMethodDef bfcm_methods[] = {
    { "spear",
      (PyCFunction)py_spear,
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
PyInit_spear (void)
{
    import_array ();
    return PyModule_Create (&moduledef);
}
