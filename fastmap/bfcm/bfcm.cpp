#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <limits>

#include "lap/lap.h"

namespace py = pybind11;

double bfcm(py::array_t<double> D, const int nv, const int nc) {
    py::buffer_info D_buf = D.request();
    if (D_buf.ndim != 4) throw std::runtime_error("Number of dimensions must be 4");
    if (D_buf.shape[0] != D_buf.shape[1] || D_buf.shape[2] != D_buf.shape[3])
        throw std::runtime_error("Shape of D must be (n,n,m,m)");
    double *ptr_D = static_cast<double *>(D_buf.ptr);

    int sigma[nc];  // Permutation array
    for (int i = 0; i < nc; i++) sigma[i] = i;

    int a[nv], b[nv];
    double cost[nv * nv], u[nv], v[nv];
    double best_res = std::numeric_limits<double>::infinity();

    const int nvnv = nv * nv;
    const int ncnc = nc * nc;
    const int nvnvncnc = nvnv * ncnc;

    do {
        std::memset(cost, 0, nv * nv * sizeof(double));
        for (int k = 0; k < nc; k++)
            for (int i = 0; i < nv; i++)
                for (int j = 0; j < nv; j++)
                    cost[i * nv + j] += ptr_D[i * nv * nc * nc + j * nc * nc + k * nc + sigma[k]];

        best_res = std::min(best_res, lap(nv, cost, a, b, u, v));
    } while (std::next_permutation(sigma, sigma + nc));

    return best_res;
}

PYBIND11_MODULE(bfcm, m) {
    m.doc() = "";  // optional module docstring
    m.def("bfcm", &bfcm, "Exhaustive search over all possible candidates matchings");
}