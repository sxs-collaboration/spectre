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
class BoundaryCondition
    : public virtual domain::BoundaryConditions::BoundaryCondition {
 public:
  BoundaryCondition() = default;
  BoundaryCondition(BoundaryCondition&&) = default;
  BoundaryCondition& operator=(BoundaryCondition&&) = default;
  BoundaryCondition(const BoundaryCondition&) = default;
  BoundaryCondition& operator=(const BoundaryCondition&) = default;
  ~BoundaryCondition() override = default;

  void pup([[maybe_unused]] PUP::er& p) override {
#if defined(SPECTRE_USE_CHARM)
    domain::BoundaryConditions::BoundaryCondition::pup(p);
#endif  // SPECTRE_USE_CHARM
  }
};
}  // namespace BoundaryConditions
}  // namespace RadiationTransport::M1Grey
