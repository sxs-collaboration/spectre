// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ForceFree/ElectricCurrentDensity.hpp"
#include "Evolution/Systems/ForceFree/Fluxes.hpp"
#include "Evolution/Systems/ForceFree/Sources.hpp"
#include "Evolution/Systems/ForceFree/TimeDerivativeTerms.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/Gsl.hpp"

namespace ForceFree {

void TimeDerivativeTerms::apply(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        non_flux_terms_dt_tilde_e,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        non_flux_terms_dt_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_tilde_psi,
    const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> /*non_flux_terms_dt_tilde_q*/,

    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_e_flux,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_psi_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_phi_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_q_flux,

    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        lapse_times_electric_field_one_form,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        lapse_times_magnetic_field_one_form,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_j_drift,
    const gsl::not_null<tnsr::ijj<DataVector, 3, Frame::Inertial>*>
        spatial_christoffel_first_kind,
    const gsl::not_null<tnsr::Ijj<DataVector, 3, Frame::Inertial>*>
        spatial_christoffel_second_kind,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        trace_spatial_christoffel_second,

    const gsl::not_null<Scalar<DataVector>*> temp_lapse,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> temp_shift,
    const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
        temp_inverse_spatial_metric,

    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_psi, const Scalar<DataVector>& tilde_phi,
    const Scalar<DataVector>& tilde_q,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_j,
    const double kappa_psi, const double kappa_phi,
    const double parallel_conductivity,

    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& extrinsic_curvature,
    const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
    const tnsr::ijj<DataVector, 3, Frame::Inertial>& d_spatial_metric) {
  // Note that if the temp_lapse and lapse arguments point to the same object
  // then the copy is elided internally.
  *temp_lapse = lapse;
  *temp_shift = shift;
  *temp_inverse_spatial_metric = inv_spatial_metric;

  // Compute fluxes
  raise_or_lower_index(lapse_times_electric_field_one_form, tilde_e,
                       spatial_metric);
  raise_or_lower_index(lapse_times_magnetic_field_one_form, tilde_b,
                       spatial_metric);
  for (size_t d = 0; d < 3; ++d) {
    (*lapse_times_electric_field_one_form).get(d) *=
        get(lapse) / get(sqrt_det_spatial_metric);
    (*lapse_times_magnetic_field_one_form).get(d) *=
        get(lapse) / get(sqrt_det_spatial_metric);
  }
  detail::fluxes_impl(
      tilde_e_flux, tilde_b_flux, tilde_psi_flux, tilde_phi_flux, tilde_q_flux,
      *lapse_times_electric_field_one_form,
      *lapse_times_magnetic_field_one_form, tilde_e, tilde_b, tilde_psi,
      tilde_phi, tilde_q, tilde_j, lapse, shift, inv_spatial_metric);

  // Compute source terms
  ComputeDriftTildeJ::apply(tilde_j_drift, tilde_q, tilde_e, tilde_b,
                            parallel_conductivity, lapse,
                            sqrt_det_spatial_metric, spatial_metric);
  gr::christoffel_first_kind(spatial_christoffel_first_kind, d_spatial_metric);
  raise_or_lower_first_index(spatial_christoffel_second_kind,
                             *spatial_christoffel_first_kind,
                             inv_spatial_metric);
  trace_last_indices(trace_spatial_christoffel_second,
                     *spatial_christoffel_second_kind, inv_spatial_metric);

  detail::sources_impl(non_flux_terms_dt_tilde_e, non_flux_terms_dt_tilde_b,
                       non_flux_terms_dt_tilde_psi, non_flux_terms_dt_tilde_phi,
                       *trace_spatial_christoffel_second, tilde_e, tilde_b,
                       tilde_psi, tilde_phi, tilde_q, *tilde_j_drift, kappa_psi,
                       kappa_phi, lapse, d_lapse, d_shift, inv_spatial_metric,
                       extrinsic_curvature);
}

}  // namespace ForceFree
