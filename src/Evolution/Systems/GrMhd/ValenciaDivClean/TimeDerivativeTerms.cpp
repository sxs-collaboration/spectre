// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/TimeDerivativeTerms.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean {
namespace detail {
void fluxes_impl(
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_d_flux,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_tau_flux,
    gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> tilde_s_flux,
    gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_phi_flux,

    // Temporaries
    gsl::not_null<Scalar<DataVector>*> transport_velocity,
    const tnsr::i<DataVector, 3, Frame::Inertial>& lapse_b_over_w,
    const Scalar<DataVector>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataVector>& pressure_star_lapse_sqrt_det_spatial_metric,

    // Extra args
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity) noexcept;

void sources_impl(
    gsl::not_null<Scalar<DataVector>*> source_tilde_tau,
    gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> source_tilde_s,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> source_tilde_b,
    gsl::not_null<Scalar<DataVector>*> source_tilde_phi,

    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_s_up,
    gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*> densitized_stress,
    gsl::not_null<Scalar<DataVector>*> h_rho_w_squared_plus_b_squared,

    const Scalar<DataVector>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataVector>& magnetic_field_squared,
    const Scalar<DataVector>& one_over_w_squared,
    const Scalar<DataVector>& pressure_star,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        trace_spatial_christoffel_second,

    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
    const tnsr::ijj<DataVector, 3, Frame::Inertial>& d_spatial_metric,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,

    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& extrinsic_curvature,
    double constraint_damping_parameter) noexcept;
}  // namespace detail

void TimeDerivativeTerms::apply(
    const gsl::not_null<Scalar<DataVector>*> /*non_flux_terms_dt_tilde_d*/,
    const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        non_flux_terms_dt_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        non_flux_terms_dt_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_tilde_phi,

    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_d_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_tau_flux,
    const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> tilde_s_flux,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_phi_flux,

    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        spatial_velocity_one_form,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        magnetic_field_one_form,
    const gsl::not_null<Scalar<DataVector>*>
        magnetic_field_dot_spatial_velocity,
    const gsl::not_null<Scalar<DataVector>*> magnetic_field_squared,
    const gsl::not_null<Scalar<DataVector>*> one_over_w_squared,
    const gsl::not_null<Scalar<DataVector>*> pressure_star,
    const gsl::not_null<Scalar<DataVector>*>
        pressure_star_lapse_sqrt_det_spatial_metric,
    const gsl::not_null<Scalar<DataVector>*> transport_velocity,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        lapse_b_over_w,

    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_s_up,
    const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
        densitized_stress,
    const gsl::not_null<tnsr::ijj<DataVector, 3, Frame::Inertial>*>
        spatial_christoffel_first_kind,
    const gsl::not_null<tnsr::Ijj<DataVector, 3, Frame::Inertial>*>
        spatial_christoffel_second_kind,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        trace_spatial_christoffel_second,
    const gsl::not_null<Scalar<DataVector>*> h_rho_w_squared_plus_b_squared,

    const gsl::not_null<Scalar<DataVector>*> temp_lapse,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> temp_shift,
    const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
        temp_inverse_spatial_metric,

    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
    const tnsr::ijj<DataVector, 3, Frame::Inertial>& d_spatial_metric,
    const Scalar<DataVector>& pressure,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,

    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& extrinsic_curvature,
    const double constraint_damping_parameter) noexcept {
  // Note that if the temp_lapse and lapse arguments point to the same object
  // then the copy is elided internally.
  *temp_lapse = lapse;
  *temp_shift = shift;
  *temp_inverse_spatial_metric = inv_spatial_metric;

  raise_or_lower_index(spatial_velocity_one_form, spatial_velocity,
                       spatial_metric);
  raise_or_lower_index(magnetic_field_one_form, magnetic_field, spatial_metric);
  dot_product(magnetic_field_dot_spatial_velocity, magnetic_field,
              *spatial_velocity_one_form);
  dot_product(magnetic_field_squared, magnetic_field, *magnetic_field_one_form);
  get(*one_over_w_squared) = 1.0 / square(get(lorentz_factor));
  get(*pressure_star) =
      get(pressure) + 0.5 * square(get(*magnetic_field_dot_spatial_velocity)) +
      0.5 * get(*magnetic_field_squared) * get(*one_over_w_squared);
  get(*pressure_star_lapse_sqrt_det_spatial_metric) =
      get(sqrt_det_spatial_metric) * get(lapse) * get(*pressure_star);

  // lapse b_i / W = lapse (B_i / W^2 + v_i (B^m v_m)
  *lapse_b_over_w = *spatial_velocity_one_form;
  for (size_t i = 0; i < 3; ++i) {
    lapse_b_over_w->get(i) *= get(*magnetic_field_dot_spatial_velocity);
    lapse_b_over_w->get(i) +=
        get(*one_over_w_squared) * magnetic_field_one_form->get(i);
    lapse_b_over_w->get(i) *= get(lapse);
  }

  detail::fluxes_impl(
      tilde_d_flux, tilde_tau_flux, tilde_s_flux, tilde_b_flux, tilde_phi_flux,
      // Temporaries
      transport_velocity, *lapse_b_over_w, *magnetic_field_dot_spatial_velocity,
      *pressure_star_lapse_sqrt_det_spatial_metric,
      // Extra args
      tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse, shift,
      inv_spatial_metric, spatial_velocity);

  // Compute source terms
  gr::christoffel_first_kind(spatial_christoffel_first_kind, d_spatial_metric);
  raise_or_lower_first_index(spatial_christoffel_second_kind,
                             *spatial_christoffel_first_kind,
                             inv_spatial_metric);
  trace_last_indices(trace_spatial_christoffel_second,
                     *spatial_christoffel_second_kind, inv_spatial_metric);

  detail::sources_impl(
      non_flux_terms_dt_tilde_tau, non_flux_terms_dt_tilde_s,
      non_flux_terms_dt_tilde_b, non_flux_terms_dt_tilde_phi,

      tilde_s_up, densitized_stress, h_rho_w_squared_plus_b_squared,

      *magnetic_field_dot_spatial_velocity, *magnetic_field_squared,
      *one_over_w_squared, *pressure_star, *trace_spatial_christoffel_second,

      tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse,
      sqrt_det_spatial_metric, inv_spatial_metric, d_lapse, d_shift,
      d_spatial_metric, spatial_velocity, lorentz_factor, magnetic_field,

      rest_mass_density, specific_enthalpy, extrinsic_curvature,
      constraint_damping_parameter);
}
}  // namespace grmhd::ValenciaDivClean
