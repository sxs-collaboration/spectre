// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Xcts/CenterOfMass.hpp"

namespace Xcts {

void center_of_mass_surface_integrand(
    gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::I<DataVector, 3>& unit_normal) {
  tenex::evaluate<ti::I>(result, 3. / (8. * M_PI) * pow<4>(conformal_factor()) *
                                     unit_normal(ti::I));
}

tnsr::I<DataVector, 3> center_of_mass_surface_integrand(
    const Scalar<DataVector>& conformal_factor,
    const tnsr::I<DataVector, 3>& unit_normal) {
  tnsr::I<DataVector, 3> result;
  center_of_mass_surface_integrand(make_not_null(&result), conformal_factor,
                                   unit_normal);
  return result;
}

}  // namespace Xcts
