// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/BoundaryConditions/None.hpp"

namespace domain::BoundaryConditions {
namespace detail {
MarkAsNone::~MarkAsNone() = default;
}  // namespace detail

bool is_none(
    const std::unique_ptr<BoundaryCondition>& boundary_condition) noexcept {
  return dynamic_cast<const detail::MarkAsNone* const>(
             boundary_condition.get()) != nullptr;
}
}  // namespace domain::BoundaryConditions
