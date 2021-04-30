// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/BoundaryConditions/RegisterDerivedWithCharm.hpp"

#include "Evolution/Systems/Burgers/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/Dirichlet.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/Outflow.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace Burgers::BoundaryConditions {
void register_derived_with_charm() noexcept {
  Parallel::register_derived_classes_with_charm<BoundaryCondition>();
}
}  // namespace Burgers::BoundaryConditions
