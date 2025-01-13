#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*
 * Function: kemeny_distance
 * -------------------------
 * Computes the Kemeny distance for a given ranking and pairwise counts.
 *
 * Parameters:
 *  - `ranking`: Pointer to the ranking array.
 *  - `n`: Number of items in the ranking.
 *  - `pairwise_counts`: 2D array representing pairwise disagreements between items.
 *
 * Returns:
 *  - int: The Kemeny distance for the given ranking.
 *
 * Complexity:
 *  - Time: O(n^2), where `n` is the size of the ranking.
 *  - Space: O(1), no additional memory allocation.
 *
 * Notes:
 *  - This function assumes valid input arrays.
 */
static inline int kemeny_distance(int* ranking, int n, int** pairwise_counts) {
    int distance = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (ranking[i] < ranking[j])
                distance += pairwise_counts[j][i];
            else
                distance += pairwise_counts[i][j];
        }
    }
    return distance;
}

/*
 * Function: swap
 * --------------
 * Swaps two elements in an array.
 *
 * Parameters:
 *  - `a`: Pointer to the first element.
 *  - `b`: Pointer to the second element.
 *
 * Complexity:
 *  - Time: O(1), constant-time operation.
 *  - Space: O(1), no additional memory allocation.
 *
 * Notes:
 *  - Swaps the values at the provided pointers.
 */
static inline void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

/*
 * Function: generate_neighbor
 * ---------------------------
 * Generates a neighboring ranking by swapping two random elements.
 *
 * Parameters:
 *  - `ranking`: Pointer to the ranking array.
 *  - `n`: Number of items in the ranking.
 *
 * Complexity:
 *  - Time: O(1), constant-time operation.
 *  - Space: O(1), no additional memory allocation.
 *
 * Notes:
 *  - Ensures the generated neighbor differs from the original ranking.
 */
static inline void generate_neighbor(int* ranking, int n) {
    int i = rand() % n;
    int j = rand() % n;
    while (i == j) {
        j = rand() % n;
    }
    swap(&ranking[i], &ranking[j]);
}

/*
 * Function: precompute_pairwise_disagreements
 * -------------------------------------------
 * Precomputes pairwise disagreements based on a set of rankings.
 *
 * Parameters:
 *  - `rankings`: 2D array of rankings.
 *  - `num_rankings`: Number of rankings in the dataset.
 *  - `n`: Number of items in each ranking.
 *  - `pairwise_counts`: 2D array to store pairwise disagreement counts.
 *
 * Complexity:
 *  - Time: O(num_rankings * n^2), where `n` is the number of items.
 *  - Space: O(n^2), memory allocated for pairwise counts.
 *
 * Notes:
 *  - Populates the `pairwise_counts` array with computed values.
 */
static inline void precompute_pairwise_disagreements(int** rankings, int num_rankings, int n, int** pairwise_counts) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            pairwise_counts[i][j] = 0;
        }
    }

    for (int r = 0; r < num_rankings; r++) {
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int index_i = -1, index_j = -1;
                for (int k = 0; k < n; k++) {
                    if (rankings[r][k] == i) {
                        index_i = k;
                    } else if (rankings[r][k] == j) {
                        index_j = k;
                    }
                    if (index_i != -1 && index_j != -1) {
                        break;
                    }
                }

                if (index_i < index_j) {
                    pairwise_counts[i][j] += 1;
                } else {
                    pairwise_counts[j][i] += 1;
                }
            }
        }
    }
}

/*
 * Function: simulated_annealing
 * -----------------------------
 * Performs simulated annealing to find the best ranking that minimizes the Kemeny distance.
 *
 * Parameters:
 *  - `rankings`: 2D array of rankings.
 *  - `num_rankings`: Number of rankings in the dataset.
 *  - `n`: Number of items in each ranking.
 *  - `initial_temp`: Initial temperature for simulated annealing.
 *  - `cooling_rate`: Rate at which the temperature decreases.
 *  - `max_iterations`: Maximum number of iterations.
 *  - `best_ranking`: Pointer to store the best ranking found.
 *
 * Returns:
 *  - int: The Kemeny distance for the best ranking found.
 *
 * Complexity:
 *  - Time: O(max_iterations * n^2), due to repeated neighbor evaluations.
 *  - Space: O(n^2), for pairwise counts storage.
 *
 * Notes:
 *  - Uses a probabilistic approach to escape local minima.
 */
static inline int simulated_annealing(int** rankings, int num_rankings, int n, double initial_temp, double cooling_rate, int max_iterations, int* best_ranking) {
    int** pairwise_counts = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        pairwise_counts[i] = (int*)malloc(n * sizeof(int));
    }
    precompute_pairwise_disagreements(rankings, num_rankings, n, pairwise_counts);

    int* current_ranking = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        current_ranking[i] = i;
        best_ranking[i] = i;
    }

    int best_distance = kemeny_distance(current_ranking, n, pairwise_counts);
    int current_distance = best_distance;

    double temperature = initial_temp;

    for (int iteration = 0; iteration < max_iterations; iteration++) {
        int* neighbor_ranking = (int*)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) {
            neighbor_ranking[i] = current_ranking[i];
        }
        generate_neighbor(neighbor_ranking, n);
        int neighbor_distance = kemeny_distance(neighbor_ranking, n, pairwise_counts);

        if (neighbor_distance < current_distance || (exp((current_distance - neighbor_distance) / temperature) > ((double)rand() / RAND_MAX))) {
            for (int i = 0; i < n; i++) {
                current_ranking[i] = neighbor_ranking[i];
            }
            current_distance = neighbor_distance;
        }

        if (current_distance < best_distance) {
            for (int i = 0; i < n; i++) {
                best_ranking[i] = current_ranking[i];
            }
            best_distance = current_distance;
        }

        temperature *= cooling_rate;
        free(neighbor_ranking);
    }

    for (int i = 0; i < n; i++) {
        free(pairwise_counts[i]);
    }
    free(pairwise_counts);
    free(current_ranking);

    return best_distance;
}


/*
 * Function: py_simulated_annealing
 * ---------------------------------
 * Python wrapper for the simulated annealing function.
 *
 * Parameters:
 *  - Python arguments:
 *      - `rankings` (list of lists): 2D list of rankings.
 *      - `n` (int): Number of items in each ranking.
 *      - `initial_temp` (float): Initial temperature.
 *      - `cooling_rate` (float): Cooling rate for temperature.
 *      - `max_iterations` (int): Maximum number of iterations.
 *
 * Returns:
 *  - list: Best ranking found.
 *
 * Notes:
 *  - Validates Python arguments and performs conversions.
 */
static PyObject *py_simulated_annealing(PyObject* self, PyObject* args) {
    PyObject* rankings_list;
    int n, max_iterations;
    double initial_temp, cooling_rate;

    if (!PyArg_ParseTuple(args, "O!idii", &PyList_Type, &rankings_list, &n, &initial_temp, &cooling_rate, &max_iterations)) {
        return NULL;
    }

    int num_rankings = PyList_Size(rankings_list);
    int** rankings = (int**)malloc(num_rankings * sizeof(int*));
    for (int i = 0; i < num_rankings; i++) {
        PyObject* ranking_row = PyList_GetItem(rankings_list, i);
        rankings[i] = (int*)malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) {
            rankings[i][j] = PyLong_AsLong(PyList_GetItem(ranking_row, j));
        }
    }

    int* best_ranking = (int*)malloc(n * sizeof(int));
    int best_distance = simulated_annealing(rankings, num_rankings, n, initial_temp, cooling_rate, max_iterations, best_ranking);

    PyObject* best_ranking_list = PyList_New(n);
    for (int i = 0; i < n; i++) {
        PyList_SetItem(best_ranking_list, i, PyLong_FromLong(best_ranking[i]));
    }

    for (int i = 0; i < num_rankings; i++) {
        free(rankings[i]);
    }
    free(rankings);
    free(best_ranking);

    return Py_BuildValue("i", best_distance);
}

/*
 * Module definition: simulated_annealing
 * ---------------------------------------
 * Provides Python bindings for the simulated annealing algorithm.
 */
static PyMethodDef SimulatedAnnealingMethods[] = {
    {"simulated_annealing", (PyCFunction)py_simulated_annealing, METH_VARARGS, "Simulated Annealing for k-Kemeny distance."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef simulatedannealingmodule = {
    PyModuleDef_HEAD_INIT,
    "_simulated_annealing",
    "Simulated Annealing Module for Kemeny distance.",
    -1,
    SimulatedAnnealingMethods
};

PyMODINIT_FUNC PyInit__simulated_annealing(void) {
    import_array();
    return PyModule_Create(&simulatedannealingmodule);
}