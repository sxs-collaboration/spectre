// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace EquationsOfState {
namespace py_bindings {
void bind_equation_of_state(py::module& m);  // NOLINT
void bind_polytropic_fluid(py::module& m);   // NOLINT
}  // namespace py_bindings
}  // namespace EquationsOfState

PYBIND11_MODULE(_PyEquationsOfState, m) {  // NOLINT
  EquationsOfState::py_bindings::bind_equation_of_state(m);
  EquationsOfState::py_bindings::bind_polytropic_fluid(m);
}
