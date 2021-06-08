// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"

#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
Hll::Hll(CkMigrateMessage* /*unused*/) noexcept {}

std::unique_ptr<BoundaryCorrection> Hll::get_clone() const noexcept {
  return std::make_unique<Hll>(*this);
}

void Hll::pup(PUP::er& p) { BoundaryCorrection::pup(p); }

double Hll::dg_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_largest_outgoing_char_speed,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_largest_ingoing_char_speed,

    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi,

    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_d,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_tau,
    const tnsr::Ij<DataVector, 3, Frame::Inertial>& flux_tilde_s,
    const tnsr::IJ<DataVector, 3, Frame::Inertial>& flux_tilde_b,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_phi,

    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,

    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>&
        normal_dot_mesh_velocity) noexcept {
  {
    // Compute max abs char speed
    Scalar<DataVector>& shift_dot_normal = *packaged_tilde_d;
    dot_product(make_not_null(&shift_dot_normal), shift, normal_covector);
    if (normal_dot_mesh_velocity.has_value()) {
      get(*packaged_largest_outgoing_char_speed) =
          get(lapse) - get(shift_dot_normal) - get(*normal_dot_mesh_velocity);
      get(*packaged_largest_ingoing_char_speed) =
          -get(lapse) - get(shift_dot_normal) - get(*normal_dot_mesh_velocity);
    } else {
      get(*packaged_largest_outgoing_char_speed) =
          get(lapse) - get(shift_dot_normal);
      get(*packaged_largest_ingoing_char_speed) =
          -get(lapse) - get(shift_dot_normal);
    }
  }

  *packaged_tilde_d = tilde_d;
  *packaged_tilde_tau = tilde_tau;
  *packaged_tilde_s = tilde_s;
  *packaged_tilde_b = tilde_b;
  *packaged_tilde_phi = tilde_phi;

  normal_dot_flux(packaged_normal_dot_flux_tilde_d, normal_covector,
                  flux_tilde_d);
  normal_dot_flux(packaged_normal_dot_flux_tilde_tau, normal_covector,
                  flux_tilde_tau);
  normal_dot_flux(packaged_normal_dot_flux_tilde_s, normal_covector,
                  flux_tilde_s);
  normal_dot_flux(packaged_normal_dot_flux_tilde_b, normal_covector,
                  flux_tilde_b);
  normal_dot_flux(packaged_normal_dot_flux_tilde_phi, normal_covector,
                  flux_tilde_phi);

  using std::max;
  return max(max(abs(get(*packaged_largest_outgoing_char_speed))),
             max(abs(get(*packaged_largest_ingoing_char_speed))));
}

void Hll::dg_boundary_terms(
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_phi,
    const Scalar<DataVector>& tilde_d_int,
    const Scalar<DataVector>& tilde_tau_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_int,
    const Scalar<DataVector>& tilde_phi_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_d_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_tau_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_s_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_b_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_phi_int,
    const Scalar<DataVector>& largest_outgoing_char_speed_int,
    const Scalar<DataVector>& largest_ingoing_char_speed_int,
    const Scalar<DataVector>& tilde_d_ext,
    const Scalar<DataVector>& tilde_tau_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_ext,
    const Scalar<DataVector>& tilde_phi_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_d_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_tau_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_s_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_b_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_phi_ext,
    const Scalar<DataVector>& largest_outgoing_char_speed_ext,
    const Scalar<DataVector>& largest_ingoing_char_speed_ext,
    const dg::Formulation dg_formulation) noexcept {
  // Allocate a temp buffer with four tags.
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>, ::Tags::TempScalar<3>>>
      temps{get(tilde_d_int).size()};

  // Determine lambda_max and lambda_min from the characteristic speeds info
  // from interior and exterior
  get(get<::Tags::TempScalar<0>>(temps)) =
      max(0., get(largest_outgoing_char_speed_int),
          -get(largest_ingoing_char_speed_ext));
  const DataVector& lambda_max = get(get<::Tags::TempScalar<0>>(temps));
  get(get<::Tags::TempScalar<1>>(temps)) =
      min(0., get(largest_ingoing_char_speed_int),
          -get(largest_outgoing_char_speed_ext));
  const DataVector& lambda_min = get(get<::Tags::TempScalar<1>>(temps));

  // Pre-compute two expressions made out of lambda_max and lambda_min
  get(get<::Tags::TempScalar<2>>(temps)) = lambda_max * lambda_min;
  const DataVector& lambdas_product = get(get<::Tags::TempScalar<2>>(temps));
  get(get<::Tags::TempScalar<3>>(temps)) = 1. / (lambda_max - lambda_min);
  const DataVector& one_over_lambda_max_minus_min =
      get(get<::Tags::TempScalar<3>>(temps));

  if (dg_formulation == dg::Formulation::WeakInertial) {
    get(*boundary_correction_tilde_d) =
        ((lambda_max * get(normal_dot_flux_tilde_d_int) +
          lambda_min * get(normal_dot_flux_tilde_d_ext)) +
         lambdas_product * (get(tilde_d_ext) - get(tilde_d_int))) *
        one_over_lambda_max_minus_min;
    get(*boundary_correction_tilde_tau) =
        ((lambda_max * get(normal_dot_flux_tilde_tau_int) +
          lambda_min * get(normal_dot_flux_tilde_tau_ext)) +
         lambdas_product * (get(tilde_tau_ext) - get(tilde_tau_int))) *
        one_over_lambda_max_minus_min;
    get(*boundary_correction_tilde_phi) =
        ((lambda_max * get(normal_dot_flux_tilde_phi_int) +
          lambda_min * get(normal_dot_flux_tilde_phi_ext)) +
         lambdas_product * (get(tilde_phi_ext) - get(tilde_phi_int))) *
        one_over_lambda_max_minus_min;

    for (size_t i = 0; i < 3; ++i) {
      boundary_correction_tilde_s->get(i) =
          ((lambda_max * normal_dot_flux_tilde_s_int.get(i) +
            lambda_min * normal_dot_flux_tilde_s_ext.get(i)) +
           lambdas_product * (tilde_s_ext.get(i) - tilde_s_int.get(i))) *
          one_over_lambda_max_minus_min;
      boundary_correction_tilde_b->get(i) =
          ((lambda_max * normal_dot_flux_tilde_b_int.get(i) +
            lambda_min * normal_dot_flux_tilde_b_ext.get(i)) +
           lambdas_product * (tilde_b_ext.get(i) - tilde_b_int.get(i))) *
          one_over_lambda_max_minus_min;
    }
  } else {
    get(*boundary_correction_tilde_d) =
        (lambda_min * (get(normal_dot_flux_tilde_d_int) +
                       get(normal_dot_flux_tilde_d_ext)) +
         lambdas_product * (get(tilde_d_ext) - get(tilde_d_int))) *
        one_over_lambda_max_minus_min;
    get(*boundary_correction_tilde_tau) =
        (lambda_min * (get(normal_dot_flux_tilde_tau_int) +
                       get(normal_dot_flux_tilde_tau_ext)) +
         lambdas_product * (get(tilde_tau_ext) - get(tilde_tau_int))) *
        one_over_lambda_max_minus_min;
    get(*boundary_correction_tilde_phi) =
        (lambda_min * (get(normal_dot_flux_tilde_phi_int) +
                       get(normal_dot_flux_tilde_phi_ext)) +
         lambdas_product * (get(tilde_phi_ext) - get(tilde_phi_int))) *
        one_over_lambda_max_minus_min;

    for (size_t i = 0; i < 3; ++i) {
      boundary_correction_tilde_s->get(i) =
          (lambda_min * (normal_dot_flux_tilde_s_int.get(i) +
                         normal_dot_flux_tilde_s_ext.get(i)) +
           lambdas_product * (tilde_s_ext.get(i) - tilde_s_int.get(i))) *
          one_over_lambda_max_minus_min;
      boundary_correction_tilde_b->get(i) =
          (lambda_min * (normal_dot_flux_tilde_b_int.get(i) +
                         normal_dot_flux_tilde_b_ext.get(i)) +
           lambdas_product * (tilde_b_ext.get(i) - tilde_b_int.get(i))) *
          one_over_lambda_max_minus_min;
    }
  }
}

// NOLINTNEXTLINE
PUP::able::PUP_ID Hll::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
