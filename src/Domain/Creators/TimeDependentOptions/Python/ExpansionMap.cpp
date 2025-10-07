// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependentOptions/Python/ExpansionMap.hpp"

#include <array>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/Creators/TimeDependentOptions/ExpansionMap.hpp"

namespace py = pybind11;

namespace domain::creators::time_dependent_options::py_bindings {
void bind_expansion_map(py::module& m) {
  py::class_<time_dependent_options::ExpansionMapOptions<false>>(
      m, "ExpansionMapOptions")
      .def(py::init<std::array<double, 3>, double, double>(),
           py::arg("initial_values_in"),
           py::arg("decay_timescale_outer_boundary_in"),
           py::arg("asymptotic_velocity_outer_boundary_in"));
}
}  // namespace domain::creators::time_dependent_options::py_bindings
