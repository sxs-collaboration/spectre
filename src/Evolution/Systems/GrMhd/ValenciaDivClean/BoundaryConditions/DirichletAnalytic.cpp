// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/DirichletAnalytic.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "Utilities/GenerateInstantiations.hpp"

namespace grmhd::ValenciaDivClean::BoundaryConditions {
DirichletAnalytic::DirichletAnalytic(CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DirichletAnalytic::get_clone() const {
  return std::make_unique<DirichletAnalytic>(*this);
}

void DirichletAnalytic::pup(PUP::er& p) { BoundaryCondition::pup(p); }

// NOLINTNEXTLINE
PUP::able::PUP_ID DirichletAnalytic::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryConditions
