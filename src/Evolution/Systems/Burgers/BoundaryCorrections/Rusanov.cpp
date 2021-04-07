// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/BoundaryCorrections/Rusanov.hpp"

#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Utilities/Gsl.hpp"

namespace Burgers::BoundaryCorrections {
Rusanov::Rusanov(CkMigrateMessage* msg) noexcept : BoundaryCorrection(msg) {}

std::unique_ptr<BoundaryCorrection> Rusanov::get_clone() const noexcept {
  return std::make_unique<Rusanov>(*this);
}

void Rusanov::pup(PUP::er& p) { BoundaryCorrection::pup(p); }

double Rusanov::dg_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_u,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_u,
    const gsl::not_null<Scalar<DataVector>*> packaged_abs_char_speed,
    const Scalar<DataVector>& u,
    const tnsr::I<DataVector, 1, Frame::Inertial>& flux_u,
    const tnsr::i<DataVector, 1, Frame::Inertial>& normal_covector,
    const std::optional<tnsr::I<DataVector, 1, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>&
        normal_dot_mesh_velocity) noexcept {
  if (normal_dot_mesh_velocity.has_value()) {
    get(*packaged_abs_char_speed) =
        abs(get(u) - get(*normal_dot_mesh_velocity));
  } else {
    get(*packaged_abs_char_speed) = abs(get(u));
  }
  *packaged_u = u;
  normal_dot_flux(packaged_normal_dot_flux_u, normal_covector, flux_u);
  return max(get(*packaged_abs_char_speed));
}

void Rusanov::dg_boundary_terms(
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_u,
    const Scalar<DataVector>& u_int,
    const Scalar<DataVector>& normal_dot_flux_u_int,
    const Scalar<DataVector>& abs_char_speed_int,
    const Scalar<DataVector>& u_ext,
    const Scalar<DataVector>& normal_dot_flux_u_ext,
    const Scalar<DataVector>& abs_char_speed_ext,
    const dg::Formulation dg_formulation) noexcept {
  if (dg_formulation == dg::Formulation::WeakInertial) {
    get(*boundary_correction_u) =
        0.5 * (get(normal_dot_flux_u_int) - get(normal_dot_flux_u_ext)) -
        0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
            (get(u_ext) - get(u_int));
  } else {
    get(*boundary_correction_u) =
        -0.5 * (get(normal_dot_flux_u_int) + get(normal_dot_flux_u_ext)) -
        0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
            (get(u_ext) - get(u_int));
  }
}
}  // namespace Burgers::BoundaryCorrections

// NOLINTNEXTLINE
PUP::able::PUP_ID Burgers::BoundaryCorrections::Rusanov::my_PUP_ID = 0;
