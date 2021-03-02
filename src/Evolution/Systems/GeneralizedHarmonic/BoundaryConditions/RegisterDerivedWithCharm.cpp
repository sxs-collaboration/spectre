// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/RegisterDerivedWithCharm.hpp"

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/DirichletAnalytic.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace GeneralizedHarmonic::BoundaryConditions {
void register_derived_with_charm() noexcept {
  Parallel::register_classes_in_list<
      typename BoundaryCondition<1>::creatable_classes>();
  Parallel::register_classes_in_list<
      typename BoundaryCondition<2>::creatable_classes>();
  Parallel::register_classes_in_list<
      typename BoundaryCondition<3>::creatable_classes>();
}
}  // namespace GeneralizedHarmonic::BoundaryConditions
