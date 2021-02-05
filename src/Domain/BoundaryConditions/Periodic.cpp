// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/BoundaryConditions/Periodic.hpp"

namespace domain::BoundaryConditions {
Periodic::~Periodic() = default;

bool is_periodic(
    const std::unique_ptr<BoundaryCondition>& boundary_condition) noexcept {
  return dynamic_cast<const Periodic* const>(boundary_condition.get()) !=
         nullptr;
}
}  // namespace domain::BoundaryConditions
