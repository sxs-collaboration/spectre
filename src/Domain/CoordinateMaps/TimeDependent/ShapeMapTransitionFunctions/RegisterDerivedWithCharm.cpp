// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/RegisterDerivedWithCharm.hpp"

#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/SphereTransition.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace domain::CoordinateMaps::ShapeMapTransitionFunctions {
void register_derived_with_charm() noexcept {
  Parallel::register_classes_with_charm<SphereTransition>();
}
}  // namespace domain::CoordinateMaps::ShapeMapTransitionFunctions
