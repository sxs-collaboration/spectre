// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"

/// \brief Boundary conditions for the Ccz4 system
namespace Ccz4::BoundaryConditions {
/// \brief The base class off of which all boundary conditions must inherit
class BoundaryCondition
    : public virtual domain::BoundaryConditions::BoundaryCondition {
 public:
  BoundaryCondition() = default;
  BoundaryCondition(BoundaryCondition&&) = default;
  BoundaryCondition& operator=(BoundaryCondition&&) = default;
  BoundaryCondition(const BoundaryCondition&) = default;
  BoundaryCondition& operator=(const BoundaryCondition&) = default;
  ~BoundaryCondition() override = default;

  void pup(PUP::er& p) override;
};
}  // namespace Ccz4::BoundaryConditions
