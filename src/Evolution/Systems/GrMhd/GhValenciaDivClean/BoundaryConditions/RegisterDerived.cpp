// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace grmhd::GhValenciaDivClean::BoundaryConditions {
void register_derived_with_charm() noexcept {
  Parallel::register_derived_classes_with_charm<BoundaryCondition>();
}
}  // namespace grmhd::GhValenciaDivClean::BoundaryConditions
