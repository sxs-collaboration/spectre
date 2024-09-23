// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Xcts/AdmLinearMomentum.hpp"

namespace Xcts {

void adm_linear_momentum_surface_integrand(
    gsl::not_null<tnsr::II<DataVector, 3>*> result,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
    const Scalar<DataVector>& trace_extrinsic_curvature) {
  tenex::evaluate<ti::I, ti::J>(
      result,
      1. / (8. * M_PI) * pow<10>(conformal_factor()) *
          (inv_extrinsic_curvature(ti::I, ti::J) -
           trace_extrinsic_curvature() * inv_spatial_metric(ti::I, ti::J)));
}

tnsr::II<DataVector, 3> adm_linear_momentum_surface_integrand(
    const Scalar<DataVector>& conformal_factor,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
    const Scalar<DataVector>& trace_extrinsic_curvature) {
  tnsr::II<DataVector, 3> result;
  adm_linear_momentum_surface_integrand(
      make_not_null(&result), conformal_factor, inv_spatial_metric,
      inv_extrinsic_curvature, trace_extrinsic_curvature);
  return result;
}

void adm_linear_momentum_volume_integrand(
    gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const tnsr::II<DataVector, 3>& surface_integrand,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3>& deriv_conformal_factor,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted) {
  // Note: we can ignore the $1/(8\pi)$ term below because it is already
  // included in `surface_integrand`.
  tenex::evaluate<ti::I>(
      result,
      -(conformal_christoffel_second_kind(ti::I, ti::j, ti::k) *
            surface_integrand(ti::J, ti::K) +
        conformal_christoffel_contracted(ti::k) *
            surface_integrand(ti::I, ti::K) -
        2. * conformal_metric(ti::j, ti::k) * surface_integrand(ti::J, ti::K) *
            inv_conformal_metric(ti::I, ti::L) * deriv_conformal_factor(ti::l) /
            conformal_factor()));
}

tnsr::I<DataVector, 3> adm_linear_momentum_volume_integrand(
    const tnsr::II<DataVector, 3>& surface_integrand,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3>& deriv_conformal_factor,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted) {
  tnsr::I<DataVector, 3> result;
  adm_linear_momentum_volume_integrand(
      make_not_null(&result), surface_integrand, conformal_factor,
      deriv_conformal_factor, conformal_metric, inv_conformal_metric,
      conformal_christoffel_second_kind, conformal_christoffel_contracted);
  return result;
}

}  // namespace Xcts
