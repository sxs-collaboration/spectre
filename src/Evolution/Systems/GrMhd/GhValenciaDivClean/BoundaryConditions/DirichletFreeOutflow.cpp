// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/DirichletFreeOutflow.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

namespace grmhd::GhValenciaDivClean::BoundaryConditions {
// LCOV_EXCL_START
DirichletFreeOutflow::DirichletFreeOutflow(CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}
// LCOV_EXCL_STOP

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DirichletFreeOutflow::get_clone() const {
  return std::make_unique<DirichletFreeOutflow>(*this);
}

void DirichletFreeOutflow::pup(PUP::er& p) { BoundaryCondition::pup(p); }

// NOLINTNEXTLINE
PUP::able::PUP_ID DirichletFreeOutflow::my_PUP_ID = 0;
}  // namespace grmhd::GhValenciaDivClean::BoundaryConditions
