#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_17_API_VERSION
#include <numpy/arrayobject.h>
#include "lap.h"

/*
Currently limited to 10 candidated per vote
*/
#define MAP_SIZE 999999999 // 9/10 already identifies a permutation of 10

uint num_of_inversions(int32_t *permutation, uint n) {
    uint res = 0;
    for (uint i = 0; i < n - 1; ++i) {
        for (uint j = i + 1; j < n; ++j) {
            if (permutation[i] > permutation[j]) {
                res++;
            }
        }
    }
    return res;
}

#define swap(type, x, y) \
    {                    \
        type SWAP = x;   \
        x = y;           \
        y = SWAP;        \
    }

int next_permutation(int32_t *p, uint m) {
    for(int i = m - 2; i >= 0; --i) {
        if (p[i] < p[i + 1]) {
	        for (int j = m - 1; j > i; --j) {
		        if (p[i] < p[j]) {
		            swap(uint, p[i], p[j]);
		            for (uint k = 1; k <= (m - i - 1) / 2; ++k) {
			            swap(uint, p[i + k], p[m - k]);
		            }
            	    return 0;
		        }
	        }
        }
    }
    return -1;
}

uint *create_id_to_inversions_map(uint candidates_num) {
    uint *lookup = malloc(sizeof(uint) * MAP_SIZE);
    int32_t *curr_permutation = malloc(sizeof(int32_t) * candidates_num);

    for(uint i = 0; i < candidates_num; ++i)
        curr_permutation[i] = i;

    do {
        uint id = 0;
        for (uint t = 0; t < candidates_num - 1; ++t) {
            id += curr_permutation[t] * pow(10, candidates_num - t - 2);
        }
        lookup[id] = num_of_inversions(curr_permutation, candidates_num);
    } while (next_permutation(curr_permutation, candidates_num) > -1);

    free(curr_permutation);
    return lookup;
}

#define free_2d_array(arr, length) \
{ \
    for (int i = 0; i < length; i++) { \
        free(arr[i]); \
    } \
    free(arr); \
}

/*
* TODO: this for sure can be done better
*/
int32_t swapDistance_election(uint votes_num, uint candidates_num, 
            const int32_t **el_one, const int32_t **el_two, const uint *lookup) {
    int32_t min_dist = 2 * candidates_num * candidates_num * votes_num;
    int32_t *mapping = malloc(candidates_num * sizeof(int32_t));
    int32_t *row_sol = malloc(votes_num * sizeof(int32_t));
    int32_t *col_sol = malloc(votes_num * sizeof(int32_t));

    int32_t *u = malloc(votes_num * sizeof(int32_t));
    int32_t *v = malloc(votes_num * sizeof(int32_t));

    int32_t *cost_matrix = malloc(votes_num * votes_num * sizeof(int32_t*));
    int32_t **e1_mapped_reversed = malloc(votes_num * sizeof(int32_t*));

    for (uint i = 0; i < votes_num; i++) {
        e1_mapped_reversed[i] = malloc(candidates_num * sizeof(int32_t));
    }
    int32_t votecomb = 0;
    for(uint i = 0; i < candidates_num; i++) { mapping[i] = i; }

    do {
        for (uint i = 0; i < votes_num; i++) {
            for (uint j = 0; j < candidates_num; j++) {
                e1_mapped_reversed[i][ mapping[ el_one[i][j] ] ] = j;
            }
        }

        for (uint i = 0; i < votes_num; i++) {
            for (uint j = 0; j < votes_num; j++) {
                votecomb = 0;
                switch(candidates_num) {
                    case 10:
                        votecomb = e1_mapped_reversed[i][el_two[j][0]] * 100000000 + e1_mapped_reversed[i][el_two[j][1]] * 10000000 
                            + e1_mapped_reversed[i][el_two[j][2]] * 1000000 +e1_mapped_reversed[i][el_two[j][3]] * 100000
                            + e1_mapped_reversed[i][el_two[j][4]] * 10000 +e1_mapped_reversed[i][el_two[j][5]] * 1000
                            + e1_mapped_reversed[i][el_two[j][6]] * 100 +e1_mapped_reversed[i][el_two[j][7]] * 10
                            + e1_mapped_reversed[i][el_two[j][8]];
                        break;
                    case 9:
                        for (int k = 0; k <= 7; k++) {
                            votecomb += e1_mapped_reversed[i][el_two[j][k]] * (int32_t)pow(10, 7 - k);
                        } break;
                    case 8:
                        for (int k = 0; k <= 6; k++) {
                            votecomb += e1_mapped_reversed[i][el_two[j][k]] * (int32_t)pow(10, 6 - k);
                        } break;
                    case 7:
                        for (int k = 0; k <= 5; k++) {
                            votecomb += e1_mapped_reversed[i][el_two[j][k]] * (int32_t)pow(10, 5 - k);
                        } break;
                    case 6:
                        for (int k = 0; k <= 4; k++) {
                            votecomb += e1_mapped_reversed[i][el_two[j][k]] * (int32_t)pow(10, 4 - k);
                        } break;
                    case 5:
                        for (int k = 0; k <= 3; k++) {
                            votecomb += e1_mapped_reversed[i][el_two[j][k]] * (int32_t)pow(10, 3 - k);
                        } break;
                    case 4:
                        for (int k = 0; k <= 2; k++) {
                            votecomb += e1_mapped_reversed[i][el_two[j][k]] * (int32_t)pow(10, 2 - k);
                        } break;
                    case 3:
                        for (int k = 0; k <= 1; k++) {
                            votecomb += e1_mapped_reversed[i][el_two[j][k]] * (int32_t)pow(10, 1 - k);
                        } break;
                }

                cost_matrix[i * votes_num + j] = lookup[votecomb];
            }
        }

        int32_t dist = lap(votes_num, cost_matrix, row_sol, col_sol, u, v);
        if (dist < min_dist) {
            min_dist = dist;
        }
    } while(next_permutation(mapping, candidates_num) > -1);

    free_2d_array(e1_mapped_reversed, votes_num);
    free(mapping);
    free(row_sol);
    free(col_sol);
    free(cost_matrix);
    free(u);
    free(v);

    return min_dist;
}

int32_t *X = NULL, *Y = NULL;
static PyObject *
py_swap (PyObject *self, PyObject *args) {
//main_swap_func(uint **el_one, uint **el_two) {
    PyObject *result = NULL, *obj_X = NULL, *obj_Y = NULL;
    int method = 0, N_METHODS = 2;
    if (!PyArg_ParseTuple (args, "OOi", &obj_X, &obj_Y, &method))
        return NULL;
    PyArrayObject *obj_cont_X = (PyArrayObject *)PyArray_ContiguousFromAny (obj_X, NPY_INT32, 0, 0);
    PyArrayObject *obj_cont_Y = (PyArrayObject *)PyArray_ContiguousFromAny (obj_Y, NPY_INT32, 0, 0);
    
    int32_t res = -1;
    if (!obj_cont_X || !obj_cont_Y)
        return NULL;

    if (method < 0 || method >= N_METHODS) {
        PyErr_Format (PyExc_ValueError, "expected method to be an int between 0 and %d", N_METHODS - 1);
        goto cleanup;
    }

    if (PyArray_NDIM (obj_cont_X) != 2 || PyArray_NDIM (obj_cont_Y) != 2) {
        PyErr_Format (PyExc_ValueError, "expected 2-D arrays, got a %d-D and a %d-D array",
                PyArray_NDIM (obj_cont_X), PyArray_NDIM (obj_cont_Y));
        goto cleanup;
    }

    X = (int32_t *) PyArray_DATA (obj_cont_X);
    Y = (int32_t *) PyArray_DATA (obj_cont_Y);

    if (X == NULL || Y == NULL) {
        PyErr_SetString(PyExc_TypeError, "invalid array object");
        goto cleanup;
    }
    npy_intp rows_X = PyArray_DIM (obj_cont_X, 0), cols_X = PyArray_DIM (obj_cont_X, 1);
    npy_intp rows_Y = PyArray_DIM (obj_cont_Y, 0), cols_Y = PyArray_DIM (obj_cont_Y, 1);
    if (rows_X != rows_Y || cols_X != cols_Y) {
        PyErr_SetString (PyExc_TypeError, "expected arrays to have the same shape");
        goto cleanup;
    }

    if (method == 1) {
        PyErr_SetString (PyExc_NotImplementedError, "Only brute force (0) so far");
        goto cleanup;
    }

    uint nc = cols_X, nv = rows_X; 
    const int32_t **a = malloc(sizeof(int32_t *) * nv);
    const int32_t **b = malloc(sizeof(int32_t *) * nv);
    for (int i = 0; i < nv; i++) {
        a[i] = X + nc * i;
        b[i] = Y + nc * i;
    }
    fflush(stdout);
    Py_BEGIN_ALLOW_THREADS;
    uint *swap_lookup = create_id_to_inversions_map(nc);
    res = swapDistance_election(nv, nc, a, b, swap_lookup);
    free(swap_lookup);
    Py_END_ALLOW_THREADS;
    free(a);
    free(b);

    result = PyLong_FromLong (res);
cleanup:
    Py_XDECREF ((PyObject *)obj_cont_X);
    Py_XDECREF ((PyObject *)obj_cont_Y);

    return result; // distance
}

static PyMethodDef methods[] = {
    { "swap", (PyCFunction)py_swap, METH_VARARGS, "Swap distance.\n" },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_swap", "Swap distance.\n", -1, methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__swap (void) {
    import_array();
    return PyModule_Create(&moduledef);
}
