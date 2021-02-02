// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/BoundaryConditions/RegisterDerivedWithCharm.hpp"

#include "Evolution/Systems/Burgers/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/Outflow.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace Burgers::BoundaryConditions {
void register_derived_with_charm() noexcept {
  Parallel::register_classes_in_list<
      typename BoundaryCondition::creatable_classes>();
}
}  // namespace Burgers::BoundaryConditions
