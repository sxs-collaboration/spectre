// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/BoundaryConditions/RegisterDerivedWithCharm.hpp"

#include "Evolution/Systems/ScalarWave/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/ScalarWave/BoundaryConditions/ConstraintPreservingSphericalRadiation.hpp"
#include "Evolution/Systems/ScalarWave/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/ScalarWave/BoundaryConditions/SphericalRadiation.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace ScalarWave::BoundaryConditions {
void register_derived_with_charm() noexcept {
  Parallel::register_derived_classes_with_charm<BoundaryCondition<1>>();
  Parallel::register_derived_classes_with_charm<BoundaryCondition<2>>();
  Parallel::register_derived_classes_with_charm<BoundaryCondition<3>>();
}
}  // namespace ScalarWave::BoundaryConditions
