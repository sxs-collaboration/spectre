// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/Python/BarycentricRational.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "NumericalAlgorithms/Interpolation/BarycentricRational.hpp"

namespace py = pybind11;

namespace intrp::py_bindings {
void bind_barycentric_rational(py::module& m) {  // NOLINT
  py::class_<BarycentricRational>(m, "BarycentricRational")
      .def(py::init<std::vector<double>, std::vector<double>, size_t>(),
           py::arg("x_values"), py::arg("y_values"), py::arg("order"))
      .def("x_values", &BarycentricRational::x_values)
      .def("y_values", &BarycentricRational::y_values)
      .def("order", &BarycentricRational::order)
      .def("__call__", &BarycentricRational::operator());
}
}  // namespace intrp::py_bindings
