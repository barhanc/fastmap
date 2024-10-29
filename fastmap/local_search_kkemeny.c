#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "kemeny.h"

static PyObject* py_local_search_kkemeny(PyObject* self, PyObject* args) {
    PyObject* votes_obj;
    int l, votes_num, cand_num;
    PyObject* starting_obj = NULL;

    // Parse the input tuple (expecting a numpy array and integers)
    if (!PyArg_ParseTuple(args, "Oiii|O", &votes_obj, &l, &votes_num, &cand_num, &starting_obj)) {
        return NULL; // If parsing fails, return NULL
    }

    // Convert the input PyObject to a numpy array for votes
    PyArrayObject* votes_array = (PyArrayObject*)PyArray_FROM_OTF(votes_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (votes_array == NULL) {
        return NULL;
    }

    int* starting = NULL;
    if (starting_obj) {
        // Convert starting to C array
        PyArrayObject* starting_array = (PyArrayObject*)PyArray_FROM_OTF(starting_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
        if (starting_array == NULL) {
            Py_DECREF(votes_array);
            return NULL;
        }
        starting = (int*)PyArray_DATA(starting_array);
        Py_DECREF(starting_array);
    }

    // Allocate and prepare the votes array
    int** votes = (int**)malloc(votes_num * sizeof(int*));
    for (int i = 0; i < votes_num; i++) {
        votes[i] = (int*)malloc(cand_num * sizeof(int));
        for (int j = 0; j < cand_num; j++) {
            votes[i][j] = *(int*)PyArray_GETPTR2(votes_array, i, j);
        }
    }

    double* result = local_search_kKemeny(votes, votes_num, cand_num, l, starting);

    // Clean up memory for votes
    for (int i = 0; i < votes_num; i++) {
        free(votes[i]);
    }
    free(votes);
    Py_DECREF(votes_array);

    // Convert result to a numpy array
    npy_intp dims[1] = {votes_num};
    PyObject* result_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, result);

    // Set result to be writable from ython
    PyArray_ENABLEFLAGS((PyArrayObject*)result_array, NPY_ARRAY_OWNDATA);

    return result_array;
}

static PyMethodDef methods[] = {
    { "local_search_kkemeny", (PyCFunction)py_local_search_kkemeny, METH_VARARGS, "Local Search Kemeny.\n" },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_local_search_kkemeny", "Kemeny module.\n", -1, methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit__local_search_kkemeny(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
