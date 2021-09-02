// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"

namespace grmhd::GhValenciaDivClean {
/// \brief Boundary conditions for the combined Generalized Harmonic and
/// Valencia GRMHD systems
namespace BoundaryConditions {

/// \brief The base class for Generalized Harmonic and Valencia combined
/// boundary conditions; all boundary conditions for this system must inherit
/// from this base class.
class BoundaryCondition : public domain::BoundaryConditions::BoundaryCondition {
 public:
  BoundaryCondition() = default;
  BoundaryCondition(BoundaryCondition&&) noexcept = default;
  BoundaryCondition& operator=(BoundaryCondition&&) noexcept = default;
  BoundaryCondition(const BoundaryCondition&) = default;
  BoundaryCondition& operator=(const BoundaryCondition&) = default;
  ~BoundaryCondition() override = default;
  explicit BoundaryCondition(CkMigrateMessage* msg) noexcept;

  void pup(PUP::er& p) override;
};
}  // namespace BoundaryConditions
}  // namespace grmhd::GhValenciaDivClean
