// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/BoundaryCorrections/Rusanov.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Utilities/Gsl.hpp"

namespace ForceFree::BoundaryCorrections {

Rusanov::Rusanov(CkMigrateMessage* msg) : BoundaryCorrection(msg) {}

std::unique_ptr<BoundaryCorrection> Rusanov::get_clone() const {
  return std::make_unique<Rusanov>(*this);
}

void Rusanov::pup(PUP::er& p) { BoundaryCorrection::pup(p); }

// NOLINTNEXTLINE
PUP::able::PUP_ID Rusanov::my_PUP_ID = 0;

double Rusanov::dg_package_data(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_tilde_e,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_psi,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_q,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_e,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_psi,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_q,
    const gsl::not_null<Scalar<DataVector>*> packaged_abs_char_speed,

    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_psi, const Scalar<DataVector>& tilde_phi,
    const Scalar<DataVector>& tilde_q,

    const tnsr::IJ<DataVector, 3, Frame::Inertial>& flux_tilde_e,
    const tnsr::IJ<DataVector, 3, Frame::Inertial>& flux_tilde_b,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_psi,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_phi,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_q,

    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,

    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity) {
  // Compute max abs char speed
  Scalar<DataVector>& shift_dot_normal = *packaged_tilde_q;
  dot_product(make_not_null(&shift_dot_normal), shift, normal_covector);
  if (normal_dot_mesh_velocity.has_value()) {
    get(*packaged_abs_char_speed) =
        max(abs(-get(lapse) - get(shift_dot_normal) -
                get(*normal_dot_mesh_velocity)),
            abs(get(lapse) - get(shift_dot_normal) -
                get(*normal_dot_mesh_velocity)));
  } else {
    get(*packaged_abs_char_speed) =
        max(abs(-get(lapse) - get(shift_dot_normal)),
            abs(get(lapse) - get(shift_dot_normal)));
  }

  *packaged_tilde_e = tilde_e;
  *packaged_tilde_b = tilde_b;
  *packaged_tilde_psi = tilde_psi;
  *packaged_tilde_phi = tilde_phi;
  *packaged_tilde_q = tilde_q;

  normal_dot_flux(packaged_normal_dot_flux_tilde_e, normal_covector,
                  flux_tilde_e);
  normal_dot_flux(packaged_normal_dot_flux_tilde_b, normal_covector,
                  flux_tilde_b);
  normal_dot_flux(packaged_normal_dot_flux_tilde_psi, normal_covector,
                  flux_tilde_psi);
  normal_dot_flux(packaged_normal_dot_flux_tilde_phi, normal_covector,
                  flux_tilde_phi);
  normal_dot_flux(packaged_normal_dot_flux_tilde_q, normal_covector,
                  flux_tilde_q);

  return max(get(*packaged_abs_char_speed));
}

void Rusanov::dg_boundary_terms(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_e,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_psi,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_q,

    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_int,
    const Scalar<DataVector>& tilde_psi_int,
    const Scalar<DataVector>& tilde_phi_int,
    const Scalar<DataVector>& tilde_q_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_e_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_b_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_psi_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_phi_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_q_int,
    const Scalar<DataVector>& abs_char_speed_int,

    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_ext,
    const Scalar<DataVector>& tilde_psi_ext,
    const Scalar<DataVector>& tilde_phi_ext,
    const Scalar<DataVector>& tilde_q_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_e_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_b_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_psi_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_phi_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_q_ext,
    const Scalar<DataVector>& abs_char_speed_ext,
    dg::Formulation dg_formulation) {
  if (dg_formulation == dg::Formulation::WeakInertial) {
    for (size_t i = 0; i < 3; ++i) {
      boundary_correction_tilde_e->get(i) =
          0.5 * (normal_dot_flux_tilde_e_int.get(i) -
                 normal_dot_flux_tilde_e_ext.get(i)) -
          0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
              (tilde_e_ext.get(i) - tilde_e_int.get(i));
      boundary_correction_tilde_b->get(i) =
          0.5 * (normal_dot_flux_tilde_b_int.get(i) -
                 normal_dot_flux_tilde_b_ext.get(i)) -
          0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
              (tilde_b_ext.get(i) - tilde_b_int.get(i));
    }
    get(*boundary_correction_tilde_psi) =
        0.5 * (get(normal_dot_flux_tilde_psi_int) -
               get(normal_dot_flux_tilde_psi_ext)) -
        0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
            (get(tilde_psi_ext) - get(tilde_psi_int));
    get(*boundary_correction_tilde_phi) =
        0.5 * (get(normal_dot_flux_tilde_phi_int) -
               get(normal_dot_flux_tilde_phi_ext)) -
        0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
            (get(tilde_phi_ext) - get(tilde_phi_int));
    get(*boundary_correction_tilde_q) =
        0.5 * (get(normal_dot_flux_tilde_q_int) -
               get(normal_dot_flux_tilde_q_ext)) -
        0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
            (get(tilde_q_ext) - get(tilde_q_int));
  } else {
    for (size_t i = 0; i < 3; ++i) {
      boundary_correction_tilde_e->get(i) =
          -0.5 * (normal_dot_flux_tilde_e_int.get(i) +
                  normal_dot_flux_tilde_e_ext.get(i)) -
          0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
              (tilde_e_ext.get(i) - tilde_e_int.get(i));
      boundary_correction_tilde_b->get(i) =
          -0.5 * (normal_dot_flux_tilde_b_int.get(i) +
                  normal_dot_flux_tilde_b_ext.get(i)) -
          0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
              (tilde_b_ext.get(i) - tilde_b_int.get(i));
    }
    get(*boundary_correction_tilde_psi) =
        -0.5 * (get(normal_dot_flux_tilde_psi_int) +
                get(normal_dot_flux_tilde_psi_ext)) -
        0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
            (get(tilde_psi_ext) - get(tilde_psi_int));
    get(*boundary_correction_tilde_phi) =
        -0.5 * (get(normal_dot_flux_tilde_phi_int) +
                get(normal_dot_flux_tilde_phi_ext)) -
        0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
            (get(tilde_phi_ext) - get(tilde_phi_int));
    get(*boundary_correction_tilde_q) =
        -0.5 * (get(normal_dot_flux_tilde_q_int) +
                get(normal_dot_flux_tilde_q_ext)) -
        0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
            (get(tilde_q_ext) - get(tilde_q_int));
  }
}

bool operator==(const Rusanov& /*lhs*/, const Rusanov& /*rhs*/) { return true; }

bool operator!=(const Rusanov& lhs, const Rusanov& rhs) {
  return not(lhs == rhs);
}

}  // namespace ForceFree::BoundaryCorrections
