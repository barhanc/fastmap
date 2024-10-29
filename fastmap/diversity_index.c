#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "kemeny.h"

void print_votes(int** votes, int num_voters, int num_candidates) {
    printf("Votes Array:\n");
    for (int i = 0; i < num_voters; i++) {
        printf("Voter %d: ", i + 1);
        for (int j = 0; j < num_candidates; j++) {
            printf("%d ", votes[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

static PyObject* py_diversity_index(PyObject* self, PyObject* args) {
    PyObject* votes_obj;
    
    if (!PyArg_ParseTuple(args, "O", &votes_obj)) {
        return NULL;
    }

    PyArrayObject* votes_array = (PyArrayObject*)PyArray_FROM_OTF(votes_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (votes_array == NULL) {
        return NULL;
    }

    int num_voters = (int)PyArray_DIM(votes_array, 0);
    int num_candidates = (int)PyArray_DIM(votes_array, 1);

    int** votes = (int**)malloc(num_voters * sizeof(int*));
    for (int i = 0; i < num_voters; ++i) {
        votes[i] = (int*)malloc(num_candidates * sizeof(int));
        for (int j = 0; j < num_candidates; ++j) {
            votes[i][j] = *(int*)PyArray_GETPTR2(votes_array, i, j);
        }
    }

    double result = diversity_index(votes, num_voters, num_candidates);

    for (int i = 0; i < num_voters; ++i) {
        free(votes[i]);
    }
    free(votes);
    Py_DECREF(votes_array);

    return Py_BuildValue("d", result);
}

static PyMethodDef methods[] = {
    { "diversity_index", (PyCFunction)py_diversity_index, METH_VARARGS, "Diversity Index.\n" },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_diversity_index", "Diversity Index.\n", -1, methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit__diversity_index (void)
{
    import_array ();
    return PyModule_Create (&moduledef);
}
