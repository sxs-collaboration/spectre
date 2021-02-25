// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/RegisterDerivedWithCharm.hpp"

#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/DirichletAnalytic.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace NewtonianEuler::BoundaryConditions {
void register_derived_with_charm() noexcept {
  Parallel::register_classes_in_list<
      typename BoundaryCondition<1>::creatable_classes>();
  Parallel::register_classes_in_list<
      typename BoundaryCondition<2>::creatable_classes>();
  Parallel::register_classes_in_list<
      typename BoundaryCondition<3>::creatable_classes>();
}
}  // namespace NewtonianEuler::BoundaryConditions
