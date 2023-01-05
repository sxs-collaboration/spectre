// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/Python/CubicSpline.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "NumericalAlgorithms/Interpolation/CubicSpline.hpp"

namespace py = pybind11;

namespace intrp::py_bindings {
void bind_cubic_spline(py::module& m) {  // NOLINT
  py::class_<CubicSpline>(m, "CubicSpline")
      .def(py::init<std::vector<double>, std::vector<double>>(),
           py::arg("x_values"), py::arg("y_values"))
      .def("x_values", &CubicSpline::x_values)
      .def("y_values", &CubicSpline::y_values)
      .def("__call__", &CubicSpline::operator());
}
}  // namespace intrp::py_bindings
