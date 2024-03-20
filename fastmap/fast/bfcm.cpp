#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <limits>

#include "lap/lap.h"

namespace py = pybind11;

double bfcm(py::array_t<double> D, const int nv, const int nc) {
    py::buffer_info D_buf = D.request();
    if (D_buf.ndim != 1) throw std::runtime_error("Number of dimensions must be 1 (use .flatten())");
    double *ptr_D = static_cast<double *>(D_buf.ptr);

    int sigma[nc];
    for (int i = 0; i < nc; i++) sigma[i] = i;

    int a[nv], b[nv];
    double cost[nv * nv], u[nv], v[nv];
    double best_res = std::numeric_limits<double>::infinity();

    const int ncnc = nc * nc;
    const int nvnv = nv * nv;
    const int nvncnc = nv * ncnc;

    do {
        std::fill(cost, cost + nvnv, 0);
        for (int i = 0; i < nv; i++)
            for (int j = 0; j < nv; j++)
                for (int k = 0; k < nc; k++) cost[i * nv + j] += ptr_D[i * nvncnc + j * ncnc + sigma[k] * nc + k];

        best_res = std::min(best_res, lap(nv, cost, a, b, u, v));
    } while (std::next_permutation(sigma, sigma + nc));

    return best_res;
}

PYBIND11_MODULE(fast, m) {
    m.doc() = "Exhaustive search over all possible candidates matchings";
    m.def("bfcm", &bfcm, "Exhaustive search over all possible candidates matchings");
}