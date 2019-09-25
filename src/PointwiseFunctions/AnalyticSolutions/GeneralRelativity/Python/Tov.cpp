// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/python.hpp>
#include <string>

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Tov.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"

namespace bp = boost::python;

namespace gr {
namespace Solutions {
namespace py_bindings {

void bind_tov() {
  bp::class_<TovSolution, boost::noncopyable>(
      "Tov",
      // boost::python receives no compile-time information on default
      // arguments, so we don't expose them to Python for now. We could write
      // thin wrappers for them if we needed to:
      // https://www.boost.org/doc/libs/1_66_0/libs/python/doc/html/tutorial/tutorial/functions.html#tutorial.functions.default_arguments
      bp::init<const EquationsOfState::EquationOfState<true, 1>&, double>(
          (bp::arg("equation_of_state"), bp::arg("central_mass_density"))))
      .def("outer_radius", &TovSolution::outer_radius)
      .def("mass_over_radius", &TovSolution::mass_over_radius)
      .def("mass", &TovSolution::mass)
      .def("log_specific_enthalpy", &TovSolution::log_specific_enthalpy);
}

}  // namespace py_bindings
}  // namespace Solutions
}  // namespace gr
