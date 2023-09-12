// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Xcts/Adm.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

#include <cmath>

#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"

namespace Xcts {

void adm_mass_volume_integrand(
    gsl::not_null<Scalar<DataVector>*> result,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const Scalar<DataVector>& conformal_ricci_scalar,
    const tnsr::ii<DataVector, 3>& extrinsic_curvature,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& christoffel_deriv) {
  tenex::evaluate(
      result,
      (1.0 / (16 * M_PI)) *
          (16.0 * M_PI * pow<5>(conformal_factor()) * energy_density() +
           (1.0 / (pow<3>(conformal_factor()))) *
               extrinsic_curvature(ti::i, ti::j) *
               extrinsic_curvature(ti::k, ti::l) *
               inv_conformal_metric(ti::I, ti::K) *
               inv_conformal_metric(ti::J, ti::L) -
           conformal_factor() * conformal_ricci_scalar() -
           pow<5>(conformal_factor()) * pow<2>(trace_extrinsic_curvature()) +
           christoffel_deriv()));
}

Scalar<DataVector> adm_mass_volume_integrand(
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inv_jacobian) {
  const tnsr::I<DataVector, 3> contracted_christoffel = tenex::evaluate<ti::I>(
      conformal_christoffel_second_kind(ti::I, ti::n, ti::j) *
      inv_conformal_metric(ti::N, ti::J));
  // const auto christoffel =
  //     partial_derivative(inv_conformal_metric,mesh,inv_jacobian);
  // const auto christoffel_contracted =
  // tenex::evaluate<ti::J>(-christoffel(ti::i,ti::I,ti::J));
  const auto christoffel_deriv =
      divergence(contracted_christoffel, mesh, inv_jacobian);
  return christoffel_deriv;
}

Scalar<DataVector> adm_mass_volume_integrand(
    const Scalar<DataVector>& conformal_factor,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const Scalar<DataVector>& conformal_ricci_scalar,
    const tnsr::ii<DataVector, 3>& extrinsic_curvature,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>& energy_density, const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inv_jacobian) {
  Scalar<DataVector> result{get_size(get(conformal_factor))};
  adm_mass_volume_integrand(
      make_not_null(&result), conformal_factor, conformal_metric,
      inv_conformal_metric, conformal_ricci_scalar, extrinsic_curvature,
      trace_extrinsic_curvature, energy_density,
      adm_mass_volume_integrand(inv_conformal_metric,
                                conformal_christoffel_second_kind, mesh,
                                inv_jacobian));
  return result;
}

void adm_mass_surface_integrand(
    gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& conformal_factor_deriv,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) {
  const auto contracted_christoffel = tenex::evaluate<ti::I>(
      conformal_christoffel_second_kind(ti::I, ti::n, ti::j) *
      inv_conformal_metric(ti::N, ti::J));
  tenex::evaluate<ti::I>(
      result, (1.0 / (16.0 * M_PI)) * (contracted_christoffel(ti::I) -
                                       8.0 * conformal_factor_deriv(ti::I)));
}

tnsr::I<DataVector, 3> adm_mass_surface_integrand(
    const tnsr::I<DataVector, 3>& conformal_factor_deriv,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) {
  tnsr::I<DataVector, 3> result{};
  adm_mass_surface_integrand(make_not_null(&result), conformal_factor_deriv,
                             inv_conformal_metric,
                             conformal_christoffel_second_kind);
  return result;
}

}  // namespace Xcts
