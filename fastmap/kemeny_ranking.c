#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "kemeny.h"

static PyObject* py_kemeny_ranking(PyObject *self, PyObject *args) {
    PyObject *obj_votes = NULL;

    // Parse the votes argument from Python
    if (!PyArg_ParseTuple(args, "O", &obj_votes)) {
        return NULL;
    }

    // Convert the input object to a contiguous NumPy array
    PyArrayObject *votes = (PyArrayObject *)PyArray_ContiguousFromAny(obj_votes, NPY_INT32, 2, 2);
    if (!votes) {
        PyErr_SetString(PyExc_ValueError, "Invalid votes array.");
        return NULL;
    }

    // Get dimensions for number of voters and candidates
    npy_intp *dims = PyArray_DIMS(votes);
    int num_voters = (int)dims[0];
    int num_candidates = (int)dims[1];

    // Create a 2D int pointer array to simulate int** from the flat data
    int **votes_array = (int **)malloc(num_voters * sizeof(int *));
    if (!votes_array) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for votes array.");
        Py_DECREF(votes);
        return NULL;
    }
    for (int i = 0; i < num_voters; i++) {
        votes_array[i] = (int *)PyArray_GETPTR2(votes, i, 0);
    }

    // Allocate memory for best ranking array and best distance
    int *best = (int *)malloc(num_candidates * sizeof(int));
    if (!best) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for best ranking array.");
        free(votes_array);
        Py_DECREF(votes);
        return NULL;
    }

    double best_d = INFINITY;

    kemeny_ranking(votes_array, num_voters, num_candidates, best, &best_d);

    PyObject *py_best = PyList_New(num_candidates);
    for (int i = 0; i < num_candidates; i++) {
        PyList_SetItem(py_best, i, PyLong_FromLong(best[i]));
    }
    PyObject *py_best_d = PyFloat_FromDouble(best_d);

    free(best);
    free(votes_array);
    Py_DECREF(votes);

    return PyTuple_Pack(2, py_best, py_best_d);
}

static PyMethodDef methods[] = {
    {"kemeny_ranking", (PyCFunction)py_kemeny_ranking, METH_VARARGS, "Calculate Kemeny ranking."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_kemeny_ranking", "Calculate Kemeny ranking.", -1, methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC 
PyInit__kemeny_ranking(void) 
{
    import_array();
    return PyModule_Create(&moduledef);
}
