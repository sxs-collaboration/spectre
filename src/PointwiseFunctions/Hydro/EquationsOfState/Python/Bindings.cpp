// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/python.hpp>

namespace EquationsOfState {
namespace py_bindings {
void bind_equation_of_state();
void bind_polytropic_fluid();
}  // namespace py_bindings
}  // namespace EquationsOfState

BOOST_PYTHON_MODULE(_PyEquationsOfState) {
  Py_Initialize();
  EquationsOfState::py_bindings::bind_equation_of_state();
  EquationsOfState::py_bindings::bind_polytropic_fluid();
}
