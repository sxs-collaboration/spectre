// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/BoundaryConditions/Dirichlet.hpp"

#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Burgers/Fluxes.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace Burgers::BoundaryConditions {
Dirichlet::Dirichlet(const double u_value) noexcept : u_value_(u_value) {}

Dirichlet::Dirichlet(CkMigrateMessage* const msg) noexcept
    : BoundaryCondition(msg) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Dirichlet::get_clone() const noexcept {
  return std::make_unique<Dirichlet>(*this);
}

void Dirichlet::pup(PUP::er& p) {
  BoundaryCondition::pup(p);
  p | u_value_;
}

std::optional<std::string> Dirichlet::dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> u,
    const gsl::not_null<tnsr::I<DataVector, 1, Frame::Inertial>*> flux_u,
    const std::optional<
        tnsr::I<DataVector, 1, Frame::Inertial>>& /*face_mesh_velocity*/,
    const tnsr::i<DataVector, 1, Frame::Inertial>& /*normal_covector*/)
    const noexcept {
  get(*u) = u_value_;
  Burgers::Fluxes::apply(flux_u, *u);
  return {};
}

// NOLINTNEXTLINE
PUP::able::PUP_ID Dirichlet::my_PUP_ID = 0;
}  // namespace Burgers::BoundaryConditions
/// \endcond
