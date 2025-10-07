// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Domain/Creators/TimeDependentOptions/Python/BinaryCompactObject.hpp"
#include "Domain/Creators/TimeDependentOptions/Python/ExpansionMap.hpp"
#include "Domain/Creators/TimeDependentOptions/Python/GridCenters.hpp"
#include "Domain/Creators/TimeDependentOptions/Python/RotationMap.hpp"
#include "Domain/Creators/TimeDependentOptions/Python/ShapeMap.hpp"
#include "Domain/Creators/TimeDependentOptions/Python/SkewMap.hpp"
#include "Domain/Creators/TimeDependentOptions/Python/TranslationMap.hpp"

namespace domain::creators::time_dependent_options {

PYBIND11_MODULE(_Pybindings, m) {
  py_bindings::bind_binary_compact_object(m);
  py_bindings::bind_expansion_map(m);
  py_bindings::bind_grid_centers(m);
  py_bindings::bind_translation_map(m);
  py_bindings::bind_skew_map(m);
  py_bindings::bind_rotation_map(m);
  py_bindings::bind_shape_map(m);
}
}  // namespace domain::creators::time_dependent_options
