// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarTensor/BoundaryConditions/ConstraintPreserving.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

namespace ScalarTensor::BoundaryConditions {
ConstraintPreserving::ConstraintPreserving(
    gh::BoundaryConditions::detail::ConstraintPreservingBjorhusType type)
    : constraint_preserving_(type) {}

// LCOV_EXCL_START
ConstraintPreserving::ConstraintPreserving(CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}
// LCOV_EXCL_STOP

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
ConstraintPreserving::get_clone() const {
  return std::make_unique<ConstraintPreserving>(*this);
}

void ConstraintPreserving::pup(PUP::er& p) {
  BoundaryCondition::pup(p);
  p | constraint_preserving_;
  p | csw_constraint_preserving_;
}

std::optional<std::string> ConstraintPreserving::dg_ghost(
    const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*>
        spacetime_metric,
    const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> pi,
    const gsl::not_null<tnsr::iaa<DataVector, 3, Frame::Inertial>*> phi,

    const gsl::not_null<Scalar<DataVector>*> psi_scalar,
    const gsl::not_null<Scalar<DataVector>*> pi_scalar,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> phi_scalar,

    // c.f. dg_package_data_temporary_tags from the combined Upwind correction
    const gsl::not_null<Scalar<DataVector>*> gamma1,
    const gsl::not_null<Scalar<DataVector>*> gamma2,
    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
    const gsl::not_null<Scalar<DataVector>*> gamma1_scalar,
    const gsl::not_null<Scalar<DataVector>*> gamma2_scalar,

    const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
        inv_spatial_metric,

    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
    /*face_mesh_velocity*/,
    const tnsr::i<DataVector, 3, Frame::Inertial>& /*normal_covector*/,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,

    const tnsr::aa<DataVector, 3, Frame::Inertial>& interior_spacetime_metric,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& interior_pi,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& interior_phi,

    const Scalar<DataVector>& psi_scalar_interior,
    const Scalar<DataVector>& pi_scalar_interior,
    const tnsr::i<DataVector, 3>& phi_scalar_interior,

    const tnsr::I<DataVector, 3, Frame::Inertial>& /*coords*/,
    const Scalar<DataVector>& interior_gamma1,
    const Scalar<DataVector>& interior_gamma2,
    const Scalar<DataVector>& interior_lapse,
    const tnsr::I<DataVector, 3>& interior_shift,
    const tnsr::II<DataVector, 3>& interior_inv_spatial_metric,
    const tnsr::AA<DataVector, 3,
                   Frame::Inertial>& /*inverse_spacetime_metric*/,
    const tnsr::A<DataVector, 3, Frame::Inertial>&
    /*spacetime_unit_normal_vector*/,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& /*three_index_constraint*/,
    const tnsr::a<DataVector, 3, Frame::Inertial>& /*gauge_source*/,
    const tnsr::ab<DataVector, 3, Frame::Inertial>&
    /*spacetime_deriv_gauge_source*/,
    const Scalar<DataVector>& interior_gamma1_scalar,
    const Scalar<DataVector>& interior_gamma2_scalar,

    // c.f. dg_interior_dt_vars_tags
    const tnsr::aa<DataVector, 3, Frame::Inertial>&
    /*logical_dt_spacetime_metric*/,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& /*logical_dt_pi*/,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& /*logical_dt_phi*/,

    const Scalar<DataVector>& /*logical_dt_psi_scalar*/,
    const Scalar<DataVector>& /*logical_dt_pi_scalar*/,
    const tnsr::i<DataVector, 3>& /*logical_dt_phi_scalar*/,

    // c.f. dg_interior_deriv_vars_tags
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& /*d_spacetime_metric*/,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& /*d_pi*/,
    const tnsr::ijaa<DataVector, 3, Frame::Inertial>& /*d_phi*/,

    const tnsr::i<DataVector, 3, Frame::Inertial>& /*d_psi_scalar*/,
    const tnsr::i<DataVector, 3, Frame::Inertial>& /*d_pi_scalar*/,
    const tnsr::ij<DataVector, 3, Frame::Inertial>& /*d_phi_scalar*/) {
  // GH
  *gamma1 = interior_gamma1;
  *gamma2 = interior_gamma2;
  *spacetime_metric = interior_spacetime_metric;
  *pi = interior_pi;
  *phi = interior_phi;
  *lapse = interior_lapse;
  *shift = interior_shift;
  *inv_spatial_metric = interior_inv_spatial_metric;

  // Scalar
  *psi_scalar = psi_scalar_interior;
  *pi_scalar = pi_scalar_interior;
  *phi_scalar = phi_scalar_interior;
  *gamma1_scalar = interior_gamma1_scalar;
  *gamma2_scalar = interior_gamma2_scalar;

  return {};
}

std::optional<std::string> ConstraintPreserving::dg_time_derivative(
    const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*>
        dt_spacetime_metric_correction,
    const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*>
        dt_pi_correction,
    const gsl::not_null<tnsr::iaa<DataVector, 3, Frame::Inertial>*>
        dt_phi_correction,

    const gsl::not_null<Scalar<DataVector>*> dt_psi_scalar_correction,
    const gsl::not_null<Scalar<DataVector>*> dt_pi_scalar_correction,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        dt_phi_scalar_correction,

    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector,
    // c.f. dg_interior_evolved_variables_tags
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& phi,

    const Scalar<DataVector>& psi_scalar,
    const Scalar<DataVector>& /*pi_scalar*/,
    const tnsr::i<DataVector, 3, Frame::Inertial>& phi_scalar,

    // c.f. dg_interior_temporary_tags
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::II<DataVector, 3>& /*interior_inv_spatial_metric*/,
    const tnsr::AA<DataVector, 3, Frame::Inertial>& inverse_spacetime_metric,
    const tnsr::A<DataVector, 3, Frame::Inertial>& spacetime_unit_normal_vector,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& three_index_constraint,
    const tnsr::a<DataVector, 3, Frame::Inertial>& gauge_source,
    const tnsr::ab<DataVector, 3, Frame::Inertial>&
        spacetime_deriv_gauge_source,
    const Scalar<DataVector>& gamma1_scalar,
    const Scalar<DataVector>& gamma2_scalar,

    // c.f. dg_interior_dt_vars_tags
    const tnsr::aa<DataVector, 3, Frame::Inertial>& logical_dt_spacetime_metric,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& logical_dt_pi,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& logical_dt_phi,

    const Scalar<DataVector>& logical_dt_psi_scalar,
    const Scalar<DataVector>& logical_dt_pi_scalar,
    const tnsr::i<DataVector, 3>& logical_dt_phi_scalar,

    // c.f. dg_interior_deriv_vars_tags
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& d_spacetime_metric,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& d_pi,
    const tnsr::ijaa<DataVector, 3, Frame::Inertial>& d_phi,

    const tnsr::i<DataVector, 3, Frame::Inertial>& d_psi_scalar,
    const tnsr::i<DataVector, 3, Frame::Inertial>& d_pi_scalar,
    const tnsr::ij<DataVector, 3, Frame::Inertial>& d_phi_scalar) const {
  // GH ConstraintPreserving
  auto gh_string = constraint_preserving_.dg_time_derivative(
      dt_spacetime_metric_correction, dt_pi_correction, dt_phi_correction,
      face_mesh_velocity, normal_covector, normal_vector, spacetime_metric, pi,
      phi, coords, gamma1, gamma2, lapse, shift, inverse_spacetime_metric,
      spacetime_unit_normal_vector, three_index_constraint, gauge_source,
      spacetime_deriv_gauge_source, logical_dt_spacetime_metric, logical_dt_pi,
      logical_dt_phi, d_spacetime_metric, d_pi, d_phi);

  // Scalar ConstraintPreservingSphericalRadiation boundary conditions
  auto scalar_string = csw_constraint_preserving_.dg_time_derivative(
      dt_psi_scalar_correction, dt_pi_scalar_correction,
      dt_phi_scalar_correction, face_mesh_velocity, normal_covector,
      normal_vector, psi_scalar, phi_scalar, coords, gamma1_scalar,
      gamma2_scalar, lapse, shift, logical_dt_psi_scalar, logical_dt_pi_scalar,
      logical_dt_phi_scalar, d_psi_scalar, d_pi_scalar, d_phi_scalar);

  if (not gh_string.has_value() and not scalar_string.has_value()) {
    return {};
  }
  if (not gh_string.has_value()) {
    return scalar_string;
  }
  if (not scalar_string.has_value()) {
    return gh_string;
  }
  return gh_string.value() + ";" + scalar_string.value();
}

// NOLINTNEXTLINE
PUP::able::PUP_ID ConstraintPreserving::my_PUP_ID = 0;
}  // namespace ScalarTensor::BoundaryConditions
