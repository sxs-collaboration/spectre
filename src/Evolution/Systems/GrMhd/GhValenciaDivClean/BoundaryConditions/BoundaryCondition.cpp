// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"

#include <pup.h>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace grmhd::GhValenciaDivClean::BoundaryConditions {
BoundaryCondition::BoundaryCondition(CkMigrateMessage* const msg) noexcept
    : domain::BoundaryConditions::BoundaryCondition(msg) {}

void BoundaryCondition::pup(PUP::er& p) {
  domain::BoundaryConditions::BoundaryCondition::pup(p);
}
}  // namespace grmhd::GhValenciaDivClean::BoundaryConditions
