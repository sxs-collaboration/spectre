// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/RegisterDerivedWithCharm.hpp"

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/DirichletAnalytic.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace grmhd::ValenciaDivClean::BoundaryConditions {
void register_derived_with_charm() noexcept {
  Parallel::register_classes_in_list<
      typename BoundaryCondition::creatable_classes>();
}
}  // namespace grmhd::ValenciaDivClean::BoundaryConditions
