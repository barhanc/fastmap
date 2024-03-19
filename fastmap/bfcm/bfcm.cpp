#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>

#include "rectangular_lsap/rectangular_lsap.h"

namespace py = pybind11;

double bfcm(py::array_t<double, py::array::f_style> D, int nv, int nc) {
    auto data = D.unchecked<4>();

    int sigma[nc] = {0};
    for (int i = 0; i < nc; i++) sigma[i] = i;

    double best_res = std::numeric_limits<double>::infinity();
    double cost[nv * nv] = {0};
    int64_t a[nv] = {0}, b[nv] = {0};

    do {
        for (int i = 0; i < nv; i++)
            for (int j = 0; j < nv; j++) {
                double acc = 0;
                for (int k = 0; k < nc; k++)
                    acc += data(i, j, k, sigma[k]);
                cost[i * nv + j] = acc;
            }

        int ret = solve_rectangular_linear_sum_assignment(nv, nv, cost, false, a, b);
        if (ret == RECTANGULAR_LSAP_INFEASIBLE || ret == RECTANGULAR_LSAP_INVALID)
            throw std::runtime_error("LSAP problem is infeasible or invalid");

        double res = 0;
        for (int i = 0; i < nv; i++)
            res += cost[a[i] * nv + b[i]];

        best_res = std::min(best_res, res);
    } while (std::next_permutation(sigma, sigma + nc));

    return best_res;
}

PYBIND11_MODULE(bfcm, m) {
    m.doc() = "";  // optional module docstring
    m.def("bf_match_cand", &bfcm, "Exhaustive search over all possible candidates matchings");
}