// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/Python/PiecewisePolytropicFluid.hpp"

#include <pybind11/pybind11.h>
#include <string>

#include "PointwiseFunctions/Hydro/EquationsOfState/PiecewisePolytropicFluid.hpp"

namespace py = pybind11;

namespace EquationsOfState::py_bindings {

void bind_piecewisepolytropic_fluid(py::module& m) {
  py::class_<PiecewisePolytropicFluid<true>, EquationOfState<true, 1>>(
      m, "RelativisticPiecewisePolytropicFluid")
      .def(py::init<double, double, double, double>(),
           py::arg("transition_density"), py::arg("polytropic_constant_lo"),
           py::arg("polytropic_exponent_lo"),
           py::arg("polytropic_exponent_hi"));
}
}  // namespace EquationsOfState::py_bindings
