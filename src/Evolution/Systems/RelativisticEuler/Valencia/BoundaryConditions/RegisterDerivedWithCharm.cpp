// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RelativisticEuler/Valencia/BoundaryConditions/RegisterDerivedWithCharm.hpp"

#include "Evolution/Systems/RelativisticEuler/Valencia/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/BoundaryConditions/DirichletAnalytic.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace RelativisticEuler::Valencia::BoundaryConditions {
void register_derived_with_charm() noexcept {
  Parallel::register_classes_in_list<
      typename BoundaryCondition<1>::creatable_classes>();
  Parallel::register_classes_in_list<
      typename BoundaryCondition<2>::creatable_classes>();
  Parallel::register_classes_in_list<
      typename BoundaryCondition<3>::creatable_classes>();
}
}  // namespace RelativisticEuler::Valencia::BoundaryConditions
