#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "kemeny.h"


static PyObject* py_polarization_index(PyObject *self, PyObject *args) {
    PyObject *obj_votes = NULL;

    if (!PyArg_ParseTuple(args, "O", &obj_votes)) {
        return NULL;
    }

    PyArrayObject *votes_array = (PyArrayObject *)PyArray_ContiguousFromAny(obj_votes, NPY_INT32, 2, 2);
    if (!votes_array) {
        PyErr_SetString(PyExc_ValueError, "Invalid votes array.");
        return NULL;
    }

    npy_intp *dims = PyArray_DIMS(votes_array);
    int num_voters = dims[0];
    int num_candidates = dims[1];

    int **votes = (int **)malloc(num_voters * sizeof(int *));
    for (int i = 0; i < num_voters; i++) {
        votes[i] = (int *)malloc(num_candidates * sizeof(int));
        memcpy(votes[i], (int *)PyArray_GETPTR2(votes_array, i, 0), num_candidates * sizeof(int)); // Copy from NumPy array
    }

    double result = polarization_index(votes, num_voters, num_candidates);

    for (int i = 0; i < num_voters; i++) {
        free(votes[i]);
    }
    free(votes);
    Py_DECREF(votes_array);

    return Py_BuildValue("d", result);
}

static PyMethodDef methods[] = {
    { "polarization_index", (PyCFunction)py_polarization_index, METH_VARARGS, "Polarization Index.\n" },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_polarization_index", "Polarization Index.\n", -1, methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit__polarization_index (void)
{
    import_array ();
    return PyModule_Create (&moduledef);
}
