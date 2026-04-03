// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"

#include <pup.h>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace grmhd::ValenciaDivClean::BoundaryConditions {
void BoundaryCondition::pup(PUP::er& p) {
#if defined(SPECTRE_USE_CHARM)
  domain::BoundaryConditions::BoundaryCondition::pup(p);
#endif  // SPECTRE_USE_CHARM
}
}  // namespace grmhd::ValenciaDivClean::BoundaryConditions
