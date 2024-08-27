// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Xcts/AdmLinearMomentum.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

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
    const tnsr::i<DataVector, 3>& conformal_factor_deriv,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted) {
  tenex::evaluate<ti::I>(
      result,
      -(conformal_christoffel_second_kind(ti::I, ti::j, ti::k) *
            surface_integrand(ti::J, ti::K) +
        conformal_christoffel_contracted(ti::k) *
            surface_integrand(ti::I, ti::K) -
        2. * conformal_metric(ti::j, ti::k) * surface_integrand(ti::J, ti::K) *
            inv_conformal_metric(ti::I, ti::L) * conformal_factor_deriv(ti::l) /
            conformal_factor()));
}

tnsr::I<DataVector, 3> adm_linear_momentum_volume_integrand(
    const tnsr::II<DataVector, 3>& surface_integrand,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3>& conformal_factor_deriv,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted) {
  tnsr::I<DataVector, 3> result;
  adm_linear_momentum_volume_integrand(
      make_not_null(&result), surface_integrand, conformal_factor,
      conformal_factor_deriv, conformal_metric, inv_conformal_metric,
      conformal_christoffel_second_kind, conformal_christoffel_contracted);
  return result;
}

}  // namespace Xcts
