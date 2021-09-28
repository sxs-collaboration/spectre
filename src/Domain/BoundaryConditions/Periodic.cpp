// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/BoundaryConditions/Periodic.hpp"

namespace domain::BoundaryConditions {
MarkAsPeriodic::~MarkAsPeriodic() = default;

bool is_periodic(const std::unique_ptr<BoundaryCondition>& boundary_condition) {
  return dynamic_cast<const MarkAsPeriodic* const>(boundary_condition.get()) !=
         nullptr;
}
}  // namespace domain::BoundaryConditions
