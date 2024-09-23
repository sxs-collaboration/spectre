// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Xcts/AdmMass.hpp"

namespace Xcts {

void adm_mass_surface_integrand(
    gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const tnsr::i<DataVector, 3>& deriv_conformal_factor,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted) {
  tenex::evaluate<ti::I>(
      result, 1. / (16. * M_PI) *
                  (inv_conformal_metric(ti::J, ti::K) *
                       conformal_christoffel_second_kind(ti::I, ti::j, ti::k) -
                   inv_conformal_metric(ti::I, ti::J) *
                       conformal_christoffel_contracted(ti::j) -
                   8. * inv_conformal_metric(ti::I, ti::J) *
                       deriv_conformal_factor(ti::j)));
}

tnsr::I<DataVector, 3> adm_mass_surface_integrand(
    const tnsr::i<DataVector, 3>& deriv_conformal_factor,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted) {
  tnsr::I<DataVector, 3> result;
  adm_mass_surface_integrand(
      make_not_null(&result), deriv_conformal_factor, inv_conformal_metric,
      conformal_christoffel_second_kind, conformal_christoffel_contracted);
  return result;
}

void adm_mass_volume_integrand(
    gsl::not_null<Scalar<DataVector>*> result,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_ricci_scalar,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
    const Scalar<DataVector>& energy_density,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::iJK<DataVector, 3>& deriv_inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
    const tnsr::iJkk<DataVector, 3>& deriv_conformal_christoffel_second_kind) {
  tenex::evaluate(
      result,
      1. / (16. * M_PI) *
          (deriv_inv_conformal_metric(ti::i, ti::J, ti::K) *
               conformal_christoffel_second_kind(ti::I, ti::j, ti::k) +
           inv_conformal_metric(ti::J, ti::K) *
               deriv_conformal_christoffel_second_kind(ti::i, ti::I, ti::j,
                                                       ti::k) +
           conformal_christoffel_contracted(ti::l) *
               inv_conformal_metric(ti::J, ti::K) *
               conformal_christoffel_second_kind(ti::L, ti::j, ti::k) -
           deriv_inv_conformal_metric(ti::i, ti::I, ti::J) *
               conformal_christoffel_contracted(ti::j) -
           inv_conformal_metric(ti::I, ti::J) *
               deriv_conformal_christoffel_second_kind(ti::i, ti::K, ti::k,
                                                       ti::j) -
           conformal_christoffel_contracted(ti::l) *
               inv_conformal_metric(ti::L, ti::J) *
               conformal_christoffel_contracted(ti::j) -
           conformal_factor() * conformal_ricci_scalar() -
           2. / 3. * pow<5>(conformal_factor()) *
               square(trace_extrinsic_curvature()) +
           pow<5>(conformal_factor()) / 4. *
              longitudinal_shift_minus_dt_conformal_metric_over_lapse_square() +
           16. * M_PI * pow<5>(conformal_factor()) * energy_density()));
}

Scalar<DataVector> adm_mass_volume_integrand(
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_ricci_scalar,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
    const Scalar<DataVector>& energy_density,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::iJK<DataVector, 3>& deriv_inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
    const tnsr::iJkk<DataVector, 3>& deriv_conformal_christoffel_second_kind) {
  Scalar<DataVector> result;
  adm_mass_volume_integrand(
      make_not_null(&result), conformal_factor, conformal_ricci_scalar,
      trace_extrinsic_curvature,
      longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
      energy_density, inv_conformal_metric, deriv_inv_conformal_metric,
      conformal_christoffel_second_kind, conformal_christoffel_contracted,
      deriv_conformal_christoffel_second_kind);
  return result;
}

}  // namespace Xcts
