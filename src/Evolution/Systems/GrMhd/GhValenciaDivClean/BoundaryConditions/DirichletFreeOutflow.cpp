// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/DirichletFreeOutflow.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

namespace grmhd::GhValenciaDivClean::BoundaryConditions {
DirichletFreeOutflow::DirichletFreeOutflow(CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}
// LCOV_EXCL_STOP
DirichletFreeOutflow::DirichletFreeOutflow(const DirichletFreeOutflow& rhs)
    : BoundaryCondition{dynamic_cast<const BoundaryCondition&>(rhs)},
      analytic_prescription_(rhs.analytic_prescription_->get_clone()) {}

DirichletFreeOutflow& DirichletFreeOutflow::operator=(
    const DirichletFreeOutflow& rhs) {
  if (&rhs == this) {
    return *this;
  }
  analytic_prescription_ = rhs.analytic_prescription_->get_clone();
  return *this;
}

DirichletFreeOutflow::DirichletFreeOutflow(
    std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription)
    : analytic_prescription_(std::move(analytic_prescription)) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DirichletFreeOutflow::get_clone() const {
  return std::make_unique<DirichletFreeOutflow>(*this);
}

void DirichletFreeOutflow::pup(PUP::er& p) {
  BoundaryCondition::pup(p);
  p | analytic_prescription_;
}
// NOLINTNEXTLINE
PUP::able::PUP_ID DirichletFreeOutflow::my_PUP_ID = 0;

}  // namespace grmhd::GhValenciaDivClean::BoundaryConditions
