// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependentOptions/Python/RotationMap.hpp"

#include <array>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/Creators/TimeDependentOptions/ExpansionMap.hpp"

namespace py = pybind11;

namespace domain::creators::time_dependent_options::py_bindings {
void bind_rotation_map(py::module& m) {
  py::class_<time_dependent_options::RotationMapOptions<false>>(
      m, "RotationMapOptions")
      .def(py::init<std::vector<std::array<double, 4>>, double>(),
           py::arg("initial_values_in"), py::arg("decay_timescale_in"));
}
}  // namespace domain::creators::time_dependent_options::py_bindings
