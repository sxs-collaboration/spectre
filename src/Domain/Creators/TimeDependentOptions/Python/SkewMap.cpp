// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependentOptions/Python/SkewMap.hpp"

#include <array>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/Creators/TimeDependentOptions/SkewMap.hpp"

namespace py = pybind11;

namespace domain::creators::time_dependent_options::py_bindings {
void bind_skew_map(py::module& m) {
  py::class_<time_dependent_options::SkewMapOptions>(m, "SkewMapOptions")
      .def(py::init<std::array<double, 3>, std::array<double, 3>>(),
           py::arg("initial_angles_y_in"), py::arg("initial_angles_z_in"));
}
}  // namespace domain::creators::time_dependent_options::py_bindings
