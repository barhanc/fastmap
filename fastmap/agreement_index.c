#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "kemeny.h"

static PyObject* py_agreement_index(PyObject *self, PyObject *args) {
    PyObject *obj_votes = NULL;

    if (!PyArg_ParseTuple(args, "O", &obj_votes)) {
        return NULL;
    }

    PyArrayObject *votes = (PyArrayObject *)PyArray_ContiguousFromAny(obj_votes, NPY_INT32, 2, 2);
    if (!votes) {
        PyErr_SetString(PyExc_ValueError, "Invalid votes array.");
        return NULL;
    }

    npy_intp *dims = PyArray_DIMS(votes);
    int num_voters = (int)dims[0];
    int num_candidates = (int)dims[1];

    int **votes_array = (int **)malloc(num_voters * sizeof(int *));
    if (!votes_array) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for votes array.");
        Py_DECREF(votes);
        return NULL;
    }
    for (int i = 0; i < num_voters; i++) {
        votes_array[i] = (int *)PyArray_GETPTR2(votes, i, 0);
    }

    double **distances = (double **)malloc(num_candidates * sizeof(double *));
    if (!distances) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for distances.");
        free(votes_array);
        Py_DECREF(votes);
        return NULL;
    }

    for (int i = 0; i < num_candidates; i++) {
        distances[i] = (double *)malloc(num_candidates * sizeof(double));
        if (!distances[i]) {
            for (int j = 0; j < i; j++) {
                free(distances[j]);
            }
            free(distances);
            free(votes_array);
            Py_DECREF(votes);
            PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for distance row.");
            return NULL;
        }
    }

    calculate_cand_dom_dist(votes_array, num_voters, num_candidates, distances);

    double sum_distances = 0;
    for (int i = 0; i < num_candidates; i++) {
        for (int j = 0; j < num_candidates; j++) {
            sum_distances += distances[i][j];
        }
    }

    for (int i = 0; i < num_candidates; i++) {
        free(distances[i]);
    }
    free(distances);
    free(votes_array);
    Py_DECREF(votes);

    double agreement_index = sum_distances / ((num_candidates - 1) * num_candidates) * 2;
    return PyFloat_FromDouble(agreement_index);
}

static PyMethodDef methods[] = {
    {"agreement_index", (PyCFunction)py_agreement_index, METH_VARARGS, "Calculate agreement index."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_agreement_index", "Calculate agreement index.", -1, methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC 
PyInit__agreement_index(void) 
{
    import_array();
    return PyModule_Create(&moduledef);
}
