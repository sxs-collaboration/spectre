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

}  // namespace Xcts
