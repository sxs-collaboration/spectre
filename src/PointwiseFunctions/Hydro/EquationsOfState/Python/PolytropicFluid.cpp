// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/Python/PolytropicFluid.hpp"

#include <pybind11/pybind11.h>
#include <string>

#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"

namespace py = pybind11;

namespace EquationsOfState::py_bindings {

void bind_polytropic_fluid(py::module& m) {
  py::class_<PolytropicFluid<true>, EquationOfState<true, 1>>(
      m, "RelativisticPolytropicFluid")
      .def(py::init<double, double>(), py::arg("polytropic_constant"),
           py::arg("polytropic_exponent"));
}

}  // namespace EquationsOfState::py_bindings
