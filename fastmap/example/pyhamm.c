#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

const int32_t *INDIC_U, *INDIC_V;
#define d(i, j, k, l) (INDIC_U[(i) * nc + (k)] ^ INDIC_V[(j) * nc + (l)])
#include "bap_bf.h"

static int32_t
hamm (const int32_t *indic_U, const int32_t *indic_V, const size_t nv, const size_t nc)
{
    INDIC_U = indic_U, INDIC_V = indic_V;
    return bap (nv, nc);
}

static PyObject *
py_hamm (PyObject *self, PyObject *args)
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
        = hamm (X, Y, nv, nc);
    Py_END_ALLOW_THREADS

        result
        = PyLong_FromLong (ret);

cleanup:
    Py_XDECREF ((PyObject *)obj_cont_X);
    Py_XDECREF ((PyObject *)obj_cont_Y);
    return result;
}

static PyMethodDef fast_methods[] = {
    { "hamm",
      (PyCFunction)py_hamm,
      METH_VARARGS,
      "Hamming distance.\n" },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "chamm",
    "Hamming distance.\n",
    -1,
    fast_methods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC
PyInit_chamm (void)
{
    import_array ();
    return PyModule_Create (&moduledef);
}
