// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/RegisterDerivedWithCharm.hpp"

#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/BoundaryCondition.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace CurvedScalarWave::BoundaryConditions {
void register_derived_with_charm() noexcept {
  Parallel::register_derived_classes_with_charm<BoundaryCondition<1>>();
  Parallel::register_derived_classes_with_charm<BoundaryCondition<2>>();
  Parallel::register_derived_classes_with_charm<BoundaryCondition<3>>();
}
}  // namespace CurvedScalarWave::BoundaryConditions
