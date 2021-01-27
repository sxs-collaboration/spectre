// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <type_traits>

#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"

namespace domain::BoundaryConditions {
namespace {
struct SystemNoBoundaryConditionsBase {};

struct BoundaryConditionsBase {};
struct SystemWithBoundaryConditionsBase {
  using boundary_conditions_base = BoundaryConditionsBase;
};

static_assert(
    not has_boundary_conditions_base_v<SystemNoBoundaryConditionsBase>);
static_assert(has_boundary_conditions_base_v<SystemWithBoundaryConditionsBase>);

static_assert(std::is_same_v<
              get_boundary_conditions_base<SystemWithBoundaryConditionsBase>,
              BoundaryConditionsBase>);
static_assert(
    std::is_same_v<get_boundary_conditions_base<SystemNoBoundaryConditionsBase>,
                   detail::TheSystemHasNoBoundaryConditionsBaseTypeAlias>);
}  // namespace
}  // namespace domain::BoundaryConditions
