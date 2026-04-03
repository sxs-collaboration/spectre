// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarTensor/BoundaryConditions/BoundaryCondition.hpp"

#include <pup.h>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace ScalarTensor::BoundaryConditions {
void BoundaryCondition::pup([[maybe_unused]] PUP::er& p) {
#if defined(SPECTRE_USE_CHARM)
  domain::BoundaryConditions::BoundaryCondition::pup(p);
#endif  // SPECTRE_USE_CHARM
}
}  // namespace ScalarTensor::BoundaryConditions
