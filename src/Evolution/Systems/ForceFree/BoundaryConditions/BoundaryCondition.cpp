// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/BoundaryConditions/BoundaryCondition.hpp"

#include <pup.h>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace ForceFree::BoundaryConditions {
void BoundaryCondition::pup(PUP::er& p) {
  domain::BoundaryConditions::BoundaryCondition::pup(p);
}
}  // namespace ForceFree::BoundaryConditions
