// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependentOptions/Python/BinaryCompactObject.hpp"

#include <array>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/Creators/TimeDependentOptions/ExpansionMap.hpp"
#include "Domain/Creators/TimeDependentOptions/GridCenters.hpp"
#include "Domain/Creators/TimeDependentOptions/RotationMap.hpp"
#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"
#include "Domain/Creators/TimeDependentOptions/SkewMap.hpp"
#include "Domain/Creators/TimeDependentOptions/TranslationMap.hpp"
#include "Domain/Structure/ObjectLabel.hpp"

namespace py = pybind11;

namespace domain::creators::time_dependent_options::py_bindings {
void bind_binary_compact_object(py::module& m) {
  py::class_<domain::creators::bco::TimeDependentMapOptions<false>>(
      m, "BinaryCompactObjectTimeDependentOptions")
      .def(py::init<
               double,
               std::optional<domain::creators::time_dependent_options::
                                 ExpansionMapOptions<false>>,
               std::optional<domain::creators::time_dependent_options::
                                 RotationMapOptions<false>>,
               std::optional<domain::creators::time_dependent_options::
                                 TranslationMapOptions<3>>,
               std::optional<
                   domain::creators::time_dependent_options::SkewMapOptions>,
               std::optional<domain::creators::time_dependent_options::
                                 ShapeMapOptions<true, ObjectLabel::A>>,
               std::optional<domain::creators::time_dependent_options::
                                 ShapeMapOptions<true, ObjectLabel::B>>,
               std::optional<domain::creators::time_dependent_options::
                                 GridCentersOptions>>(),
           py::arg("initial_time"), py::arg("expansion_map_options"),
           py::arg("rotation_map_options"), py::arg("translation_map_options"),
           py::arg("skew_map_options"), py::arg("shape_options_A"),
           py::arg("shape_options_B"), py::arg("grid_centers_options"));
}
}  // namespace domain::creators::time_dependent_options::py_bindings
