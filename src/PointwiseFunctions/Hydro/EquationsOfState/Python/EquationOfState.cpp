// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>
#include <string>

#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"

namespace py = pybind11;

namespace EquationsOfState {
namespace py_bindings {

void bind_equation_of_state(py::module& m) {  // NOLINT
  // This is a virtual base class, so we expose it only to use it in Python
  // wrappers of derived classes.
  py::class_<EquationOfState<true, 1>>(m, "RelativisticEquationOfState1D");
}

}  // namespace py_bindings
}  // namespace EquationsOfState
