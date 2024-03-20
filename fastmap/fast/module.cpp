#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <execution>
#include <iostream>
#include <limits>

#include "lap/lap.h"

namespace py = pybind11;

double bfcm(py::array_t<double> npyD, const int nv, const int nc) {
    auto D = npyD.unchecked<4>();

    double **cost = (double **)malloc(nv * sizeof(double *));
    for (int i = 0; i < nv; i++) cost[i] = (double *)malloc(nv * sizeof(double));

    for (int i = 0; i < nv; i++)
        for (int j = 0; j < nv; j++) {
            double acc = 0;
            for (int k = 0; k < nc; k++) acc += D(i, j, k, k);
            cost[i][j] = acc;
        }

    int sigma[nc], stack[nc] = {0}, alpha = 1;
    for (int i = 0; i < nc; i++) sigma[i] = i;

    int a[nv], b[nv];
    double _x[nv], _y[nv];
    double best_res = lap(nv, cost, a, b, _x, _y);

    while (alpha < nc) {
        if (stack[alpha] < alpha) {
            if (alpha % 2 == 0) {
                for (int i = 0; i < nv; i++)
                    for (int j = 0; j < nv; j++)
                        cost[i][j] += D(i, j, alpha, sigma[0]) + D(i, j, 0, sigma[alpha]) -
                                      D(i, j, 0, sigma[0]) - D(i, j, alpha, sigma[alpha]);

                std::swap(sigma[0], sigma[alpha]);

            } else {
                for (int i = 0; i < nv; i++)
                    for (int j = 0; j < nv; j++)
                        cost[i][j] += -D(i, j, alpha, sigma[alpha]) -
                                      D(i, j, stack[alpha], sigma[stack[alpha]]) +
                                      D(i, j, alpha, sigma[stack[alpha]]) +
                                      D(i, j, stack[alpha], sigma[alpha]);

                std::swap(sigma[alpha], sigma[stack[alpha]]);
            }

            best_res = std::min(best_res, lap(nv, cost, a, b, _x, _y));
            stack[alpha]++;
            alpha = 1;
        } else {
            stack[alpha] = 0;
            alpha++;
        }
    }

    for (int i = 0; i < nv; i++) free(cost[i]);
    free(cost);

    return best_res;
}

// ========================================================
// ========================================================
// ========================================================
// ========================================================
// ========================================================

PYBIND11_MODULE(fast, m) {
    m.doc() = "Exhaustive search over all possible candidates matchings";
    m.def("bfcm", &bfcm, "Exhaustive search over all possible candidates matchings");
}