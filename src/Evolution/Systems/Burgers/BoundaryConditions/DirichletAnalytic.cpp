// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/BoundaryConditions/DirichletAnalytic.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Burgers/Fluxes.hpp"

namespace Burgers::BoundaryConditions {
DirichletAnalytic::DirichletAnalytic(CkMigrateMessage* const msg) noexcept
    : BoundaryCondition(msg) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DirichletAnalytic::get_clone() const noexcept {
  return std::make_unique<DirichletAnalytic>(*this);
}

void DirichletAnalytic::pup(PUP::er& p) { BoundaryCondition::pup(p); }

void DirichletAnalytic::flux_impl(
    const gsl::not_null<tnsr::I<DataVector, 1, Frame::Inertial>*> flux,
    const Scalar<DataVector>& u_analytic) noexcept {
  Burgers::Fluxes::apply(flux, u_analytic);
}

// NOLINTNEXTLINE
PUP::able::PUP_ID DirichletAnalytic::my_PUP_ID = 0;
}  // namespace Burgers::BoundaryConditions
