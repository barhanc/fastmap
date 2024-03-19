#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdbool.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>

#include "rectangular_lsap/rectangular_lsap.h"

namespace py = pybind11;

double bfcm(py::array_t<double, py::array::f_style> D, int n_v, int n_c) {
    auto r = D.unchecked<4>();

    int sigma[n_c] = {0};
    for (int i = 0; i < n_c; i++) sigma[i] = i;

    double best_res = std::numeric_limits<double>::infinity();

    do {
        double cost[n_v * n_v] = {0};
        for (int i = 0; i < n_v; i++)
            for (int j = 0; j < n_v; j++)
                for (int k = 0; k < n_c; k++)
                    cost[i * n_v + j] += r(i, j, k, sigma[k]);

        int64_t a[n_v], b[n_v];
        int ret = solve_rectangular_linear_sum_assignment(n_v, n_v, cost, false, a, b);
        if (ret == RECTANGULAR_LSAP_INFEASIBLE || ret == RECTANGULAR_LSAP_INVALID)
            throw std::runtime_error("LSAP problem is infeasible or invalid");

        double res = 0;
        for (int i = 0; i < n_v; i++)
            res += cost[a[i] * n_v + b[i]];

        best_res = std::min(best_res, res);
    } while (std::next_permutation(sigma, sigma + n_c));

    return best_res;
}

PYBIND11_MODULE(bfcm, m) {
    m.doc() = "";  // optional module docstring
    m.def("bf_match_cand", &bfcm, "Exhaustive search over all possible candidates matchings");
}