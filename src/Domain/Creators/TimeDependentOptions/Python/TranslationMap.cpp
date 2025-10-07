// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependentOptions/Python/TranslationMap.hpp"

#include <array>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/Creators/TimeDependentOptions/TranslationMap.hpp"

namespace py = pybind11;

namespace domain::creators::time_dependent_options::py_bindings {
void bind_translation_map(py::module& m) {
  py::class_<time_dependent_options::TranslationMapOptions<3>>(
      m, "TranslationMapOptions")
      .def(py::init<std::array<std::array<double, 3>, 3>>(),
           py::arg("initial_values_in"));
}
}  // namespace domain::creators::time_dependent_options::py_bindings
