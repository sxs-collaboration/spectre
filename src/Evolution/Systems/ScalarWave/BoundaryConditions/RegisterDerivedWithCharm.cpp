// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/BoundaryConditions/RegisterDerivedWithCharm.hpp"

#include "Evolution/Systems/ScalarWave/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/ScalarWave/BoundaryConditions/DirichletAnalytic.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace ScalarWave::BoundaryConditions {
void register_derived_with_charm() noexcept {
  Parallel::register_classes_in_list<
      typename BoundaryCondition<1>::creatable_classes>();
  Parallel::register_classes_in_list<
      typename BoundaryCondition<2>::creatable_classes>();
  Parallel::register_classes_in_list<
      typename BoundaryCondition<3>::creatable_classes>();
}
}  // namespace ScalarWave::BoundaryConditions
