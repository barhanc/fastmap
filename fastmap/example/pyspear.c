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
    if (!PyArg_ParseTuple (args, "OO", &obj_X, &obj_Y))
        return NULL;

    PyArrayObject *obj_cont_X = (PyArrayObject *)PyArray_ContiguousFromAny (obj_X, NPY_INT, 0, 0);
    PyArrayObject *obj_cont_Y = (PyArrayObject *)PyArray_ContiguousFromAny (obj_Y, NPY_INT, 0, 0);
    if (!obj_cont_X || !obj_cont_Y)
        return NULL;

    X = (int32_t *)PyArray_DATA (obj_cont_X);
    Y = (int32_t *)PyArray_DATA (obj_cont_Y);
    if (X == NULL || Y == NULL)
    {
        PyErr_SetString (PyExc_TypeError, "invalid object");
        goto cleanup;
    }

    size_t nv = PyArray_DIM (obj_cont_X, 0), nc = PyArray_DIM (obj_cont_X, 1);
    int32_t ret;
    Py_BEGIN_ALLOW_THREADS
        ret
        = bap_bf (nv, nc);
    Py_END_ALLOW_THREADS

        result
        = PyLong_FromLong (ret);

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
