// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "PointwiseFunctions/Hydro/EquationsOfState/Python/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Python/PiecewisePolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Python/PolytropicFluid.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

namespace EquationsOfState {

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");
  // Abstract base class
  py_bindings::bind_equation_of_state(m);
  // Derived classes
  py_bindings::bind_piecewisepolytropic_fluid(m);
  py_bindings::bind_polytropic_fluid(m);
}

}  // namespace EquationsOfState
