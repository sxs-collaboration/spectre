// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/Python/Spectral.hpp"

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "PointwiseFunctions/Hydro/EquationsOfState/Spectral.hpp"

namespace py = pybind11;

namespace EquationsOfState::py_bindings {

void bind_spectral(py::module& m) {
  py::class_<Spectral, EquationOfState<true, 1>>(
      m, "Spectral")
      .def(py::init<double, double, std::vector<double>, double>(),
           py::arg("reference_density"), py::arg("reference_pressure"),
           py::arg("spectral_coefficients"),
           py::arg("upper_density"));
}
}  // namespace EquationsOfState::py_bindings
