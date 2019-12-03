// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/python.hpp>
#include <string>

#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"

namespace bp = boost::python;

namespace EquationsOfState {
namespace py_bindings {

void bind_polytropic_fluid() {
  // We can't expose any member functions without wrapping tensors in Python,
  // so we only expose the initializer for now.
  bp::class_<PolytropicFluid<true>, bp::bases<EquationOfState<true, 1>>>(
      "RelativisticPolytropicFluid",
      bp::init<double, double>(
          (bp::arg("polytropic_constant"), bp::arg("polytropic_exponent"))));
}

}  // namespace py_bindings
}  // namespace EquationsOfState
