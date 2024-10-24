// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/ConstraintPreservingFreeOutflow.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

namespace grmhd::GhValenciaDivClean::BoundaryConditions {
ConstraintPreservingFreeOutflow::ConstraintPreservingFreeOutflow(
    gh::BoundaryConditions::detail::ConstraintPreservingBjorhusType type)
    : constraint_preserving_(type) {}

// LCOV_EXCL_START
ConstraintPreservingFreeOutflow::ConstraintPreservingFreeOutflow(
    CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}
// LCOV_EXCL_STOP

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
ConstraintPreservingFreeOutflow::get_clone() const {
  return std::make_unique<ConstraintPreservingFreeOutflow>(*this);
}

void ConstraintPreservingFreeOutflow::pup(PUP::er& p) {
  BoundaryCondition::pup(p);
  p | constraint_preserving_;
}

std::optional<std::string> ConstraintPreservingFreeOutflow::dg_ghost(
    const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*>
        spacetime_metric,
    const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> pi,
    const gsl::not_null<tnsr::iaa<DataVector, 3, Frame::Inertial>*> phi,
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi,

    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_d_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_ye_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_tau_flux,
    const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> tilde_s_flux,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_phi_flux,

    const gsl::not_null<Scalar<DataVector>*> gamma1,
    const gsl::not_null<Scalar<DataVector>*> gamma2,
    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        spatial_velocity_one_form,
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        spatial_velocity,
    const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
        inv_spatial_metric,

    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector,

    const tnsr::aa<DataVector, 3, Frame::Inertial>& interior_spacetime_metric,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& interior_pi,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& interior_phi,

    const Scalar<DataVector>& interior_rest_mass_density,
    const Scalar<DataVector>& interior_electron_fraction,
    const Scalar<DataVector>& interior_specific_internal_energy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
    const Scalar<DataVector>& interior_lorentz_factor,
    const Scalar<DataVector>& interior_pressure,
    const Scalar<DataVector>& interior_temperature,

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

    // c.f. dg_interior_dt_vars_tags
    const tnsr::aa<DataVector, 3, Frame::Inertial>&
    /*logical_dt_spacetime_metric*/,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& /*logical_dt_pi*/,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& /*logical_dt_phi*/,
    // c.f. dg_interior_deriv_vars_tags
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& /*d_spacetime_metric*/,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& /*d_pi*/,
    const tnsr::ijaa<DataVector, 3, Frame::Inertial>& /*d_phi*/) {
  *gamma1 = interior_gamma1;
  *gamma2 = interior_gamma2;
  *spacetime_metric = interior_spacetime_metric;
  *pi = interior_pi;
  *phi = interior_phi;
  *lapse = interior_lapse;
  *shift = interior_shift;
  *inv_spatial_metric = interior_inv_spatial_metric;

  return grmhd::ValenciaDivClean::BoundaryConditions::HydroFreeOutflow::
      dg_ghost(tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi,

               tilde_d_flux, tilde_ye_flux, tilde_tau_flux, tilde_s_flux,
               tilde_b_flux, tilde_phi_flux,

               lapse, shift, spatial_velocity_one_form, rest_mass_density,
               electron_fraction, temperature, spatial_velocity,
               inv_spatial_metric,

               face_mesh_velocity, normal_covector, normal_vector,

               interior_rest_mass_density, interior_electron_fraction,
               interior_specific_internal_energy, interior_spatial_velocity,
               interior_magnetic_field, interior_lorentz_factor,
               interior_pressure, interior_temperature,

               *shift, *lapse, *inv_spatial_metric);
}

std::optional<std::string> ConstraintPreservingFreeOutflow::dg_time_derivative(
    gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*>
        dt_spacetime_metric_correction,
    gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> dt_pi_correction,
    gsl::not_null<tnsr::iaa<DataVector, 3, Frame::Inertial>*> dt_phi_correction,
    const gsl::not_null<Scalar<DataVector>*> dt_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> dt_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> dt_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> dt_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> dt_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> dt_tilde_phi,

    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector,
    // c.f. dg_interior_evolved_variables_tags
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& phi,
    // c.f. dg_interior_primitive_variables_tags
    const Scalar<DataVector>& /*interior_rest_mass_density*/,
    const Scalar<DataVector>& /*interior_electron_fraction*/,
    const Scalar<DataVector>& /*interior_specific_internal_energy*/,
    const tnsr::I<DataVector, 3,
                  Frame::Inertial>& /*interior_spatial_velocity*/,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*interior_magnetic_field*/,
    const Scalar<DataVector>& /*interior_lorentz_factor*/,
    const Scalar<DataVector>& /*interior_pressure*/,
    const Scalar<DataVector>& /*interior_temperature*/,

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
    // c.f. dg_interior_dt_vars_tags
    const tnsr::aa<DataVector, 3, Frame::Inertial>& logical_dt_spacetime_metric,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& logical_dt_pi,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& logical_dt_phi,
    // c.f. dg_interior_deriv_vars_tags
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& d_spacetime_metric,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& d_pi,
    const tnsr::ijaa<DataVector, 3, Frame::Inertial>& d_phi) const {
  get(*dt_tilde_d) = 0.0;
  get(*dt_tilde_ye) = 0.0;
  get(*dt_tilde_tau) = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    dt_tilde_s->get(i) = 0.0;
    dt_tilde_b->get(i) = 0.0;
  }
  get(*dt_tilde_phi) = 0.0;
  return constraint_preserving_.dg_time_derivative(
      dt_spacetime_metric_correction, dt_pi_correction, dt_phi_correction,
      face_mesh_velocity, normal_covector, normal_vector, spacetime_metric, pi,
      phi, coords, gamma1, gamma2, lapse, shift, inverse_spacetime_metric,
      spacetime_unit_normal_vector, three_index_constraint, gauge_source,
      spacetime_deriv_gauge_source, logical_dt_spacetime_metric, logical_dt_pi,
      logical_dt_phi, d_spacetime_metric, d_pi, d_phi);
}

// NOLINTNEXTLINE
PUP::able::PUP_ID ConstraintPreservingFreeOutflow::my_PUP_ID = 0;
}  // namespace grmhd::GhValenciaDivClean::BoundaryConditions
