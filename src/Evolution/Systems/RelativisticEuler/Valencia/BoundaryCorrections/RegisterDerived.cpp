// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RelativisticEuler/Valencia/BoundaryCorrections/Factory.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace RelativisticEuler::Valencia::BoundaryCorrections {
void register_derived_with_charm() {
  Parallel::register_derived_classes_with_charm<BoundaryCorrection<1>>();
  Parallel::register_derived_classes_with_charm<BoundaryCorrection<2>>();
  Parallel::register_derived_classes_with_charm<BoundaryCorrection<3>>();
}
}  // namespace RelativisticEuler::Valencia::BoundaryCorrections
