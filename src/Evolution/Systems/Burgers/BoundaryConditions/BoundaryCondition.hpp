// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"

namespace Burgers {
/// \brief Boundary conditions for the Burgers system
namespace BoundaryConditions {
/// \brief The base class off of which all boundary conditions must inherit
class BoundaryCondition : public domain::BoundaryConditions::BoundaryCondition {
 public:
  BoundaryCondition() = default;
  BoundaryCondition(BoundaryCondition&&) = default;
  BoundaryCondition& operator=(BoundaryCondition&&) = default;
  BoundaryCondition(const BoundaryCondition&) = default;
  BoundaryCondition& operator=(const BoundaryCondition&) = default;
  ~BoundaryCondition() override = default;

  explicit BoundaryCondition(CkMigrateMessage* msg);

  void pup(PUP::er& p) override;
};
}  // namespace BoundaryConditions
}  // namespace Burgers
