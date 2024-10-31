#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "kemeny.h"

static PyObject* py_polarization_1by2Kemenys(PyObject* self, PyObject* args) {
    PyObject* votes_obj;

    if (!PyArg_ParseTuple(args, "O", &votes_obj)) {
        return NULL;
    }

    PyArrayObject* votes_array = (PyArrayObject*)PyArray_FROM_OTF(votes_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (votes_array == NULL) {
        return NULL;
    }

    int num_voters = PyArray_DIM(votes_array, 0);
    int num_candidates = PyArray_DIM(votes_array, 1);

    int** votes = (int**)malloc(num_voters * sizeof(int*));
    for (int i = 0; i < num_voters; i++) {
        votes[i] = (int*)malloc(num_candidates * sizeof(int));
        for (int j = 0; j < num_candidates; j++) {
            votes[i][j] = *(int*)PyArray_GETPTR2(votes_array, i, j);
        }
    }

    double result = polarization_1by2Kemenys(votes, num_voters, num_candidates);

    for (int i = 0; i < num_voters; i++) {
        free(votes[i]);
    }
    free(votes);
    Py_DECREF(votes_array);

    return Py_BuildValue("d", result);
}

static PyMethodDef methods[] = {
    { "polarization_1by2Kemenys", (PyCFunction)py_polarization_1by2Kemenys, METH_VARARGS, "Calculates polarization between two Kemeny rankings." },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_polarization_1by2Kemenys", "Polarization module.\n", -1, methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__polarization_1by2Kemenys(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
