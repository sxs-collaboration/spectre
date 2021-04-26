// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/RegisterDerivedWithCharm.hpp"

#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/DirichletAnalytic.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace NewtonianEuler::BoundaryConditions {
void register_derived_with_charm() noexcept {
  Parallel::register_derived_classes_with_charm<BoundaryCondition<1>>();
  Parallel::register_derived_classes_with_charm<BoundaryCondition<2>>();
  Parallel::register_derived_classes_with_charm<BoundaryCondition<3>>();
}
}  // namespace NewtonianEuler::BoundaryConditions
