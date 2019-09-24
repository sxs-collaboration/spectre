// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/python.hpp>
#include <string>

#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"

namespace bp = boost::python;

namespace EquationsOfState {
namespace py_bindings {

void bind_equation_of_state() {
  // This is a virtual base class, so we expose it only to use it in Python
  // wrappers of derived classes.
  bp::class_<EquationOfState<true, 1>, boost::noncopyable>(
      "RelativisticEquationOfState1D", bp::no_init);
}

}  // namespace py_bindings
}  // namespace EquationsOfState
