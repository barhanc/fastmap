#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <time.h>
#include "kemeny.h"

static PyObject* py_local_search_kkemeny_single_k(PyObject* self, PyObject* args) {
    PyObject* votes_obj;
    int k, l, votes_num, cand_num;
    PyObject* starting_obj = NULL;

    // Parse the input tuple (expecting a numpy array and integers)
    if (!PyArg_ParseTuple(args, "Oiiii|O", &votes_obj, &k, &l, &votes_num, &cand_num, &starting_obj)) {
        return NULL;
    }

    // Convert the input PyObject to a numpy array for votes
    PyArrayObject* votes_array = (PyArrayObject*)PyArray_FROM_OTF(votes_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (votes_array == NULL) {
        return NULL;
    }

    // Prepare the starting array if provided, else allocate default starting
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

    int result = local_search_kKemeny_single_k(votes, k, l, votes_num, cand_num, starting);

    for (int i = 0; i < votes_num; i++) {
        free(votes[i]);
    }
    free(votes);
    Py_DECREF(votes_array);

    return Py_BuildValue("i", result);
}

static PyMethodDef methods[] = {
    { "local_search_kkemeny_single_k", (PyCFunction)py_local_search_kkemeny_single_k, METH_VARARGS, "Local Search Kemeny Single K.\n" },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_local_search_kkemeny_single_k", "Kemeny module.\n", -1, methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit__local_search_kkemeny_single_k(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
