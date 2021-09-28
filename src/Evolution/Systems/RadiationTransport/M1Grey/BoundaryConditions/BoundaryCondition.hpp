// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"

namespace RadiationTransport::M1Grey {
/// \brief Boundary conditions for the M1Grey radiation transport system
namespace BoundaryConditions {
/// \brief The base class off of which all boundary conditions must inherit
template <typename NeutrinoSpeciesList>
class BoundaryCondition : public domain::BoundaryConditions::BoundaryCondition {
 public:
  BoundaryCondition() = default;
  BoundaryCondition(BoundaryCondition&&) = default;
  BoundaryCondition& operator=(BoundaryCondition&&) = default;
  BoundaryCondition(const BoundaryCondition&) = default;
  BoundaryCondition& operator=(const BoundaryCondition&) = default;
  ~BoundaryCondition() override = default;

  explicit BoundaryCondition(CkMigrateMessage* const msg)
      : domain::BoundaryConditions::BoundaryCondition(msg) {}

  void pup(PUP::er& p) override {
    domain::BoundaryConditions::BoundaryCondition::pup(p);
  }
};
}  // namespace BoundaryConditions
}  // namespace RadiationTransport::M1Grey
