// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "PointwiseFunctions/Hydro/EquationsOfState/Python/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Python/PiecewisePolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Python/PolytropicFluid.hpp"

PYBIND11_MODULE(_PyEquationsOfState, m) {  // NOLINT
  EquationsOfState::py_bindings::bind_equation_of_state(m);
  EquationsOfState::py_bindings::bind_polytropic_fluid(m);
}
