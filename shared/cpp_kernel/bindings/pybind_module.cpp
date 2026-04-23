#include <pybind11/pybind11.h>
#include "black_scholes.h"

namespace py = pybind11;

PYBIND11_MODULE(pricing_kernel, m) {
    m.doc() = "C++ Black-Scholes pricing kernel (pybind11)";

    m.def("bs_price", &bs::bs_price,
          py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("q"), py::arg("sigma"),
          py::arg("is_call") = true);

    m.def("bs_delta", &bs::bs_delta,
          py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("q"), py::arg("sigma"),
          py::arg("is_call") = true);

    m.def("bs_gamma", &bs::bs_gamma,
          py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("q"), py::arg("sigma"));

    m.def("bs_vega", &bs::bs_vega,
          py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("q"), py::arg("sigma"));

    m.def("bs_theta", &bs::bs_theta,
          py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("q"), py::arg("sigma"),
          py::arg("is_call") = true);

    m.def("bs_implied_vol", &bs::bs_implied_vol,
          py::arg("price"), py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("q"),
          py::arg("is_call") = true);
}
