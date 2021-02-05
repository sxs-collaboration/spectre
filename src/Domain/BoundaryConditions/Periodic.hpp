// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::BoundaryConditions {
/// Mark a boundary condition as being periodic.
///
/// Periodic boundary conditions shouldn't require any implementation outside of
/// a check in the domain creator using the `is_periodic()` function to
/// determine what boundaries are periodic. Across each matching pair of
/// periodic boundary conditions, the domain creator should specify that the DG
/// elements are neighbors of each other.
class Periodic {
 public:
  Periodic() = default;
  Periodic(Periodic&&) noexcept = default;
  Periodic& operator=(Periodic&&) noexcept = default;
  Periodic(const Periodic&) = default;
  Periodic& operator=(const Periodic&) = default;
  virtual ~Periodic() = 0;
};

/// Check if a boundary condition inherits from `Periodic`, which constitutes as
/// it being marked as a periodic boundary condition.
bool is_periodic(
    const std::unique_ptr<BoundaryCondition>& boundary_condition) noexcept;
}  // namespace domain::BoundaryConditions
