// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Xcts/CenterOfMass.hpp"

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"

namespace Xcts {

void center_of_mass_surface_integrand(
    gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::I<DataVector, 3>& coords) {
  const auto euclidean_radius = magnitude(coords);
  tenex::evaluate<ti::I>(result, 3. / (8. * M_PI) * pow<4>(conformal_factor()) *
                                     coords(ti::I) / euclidean_radius());
}

tnsr::I<DataVector, 3> center_of_mass_surface_integrand(
    const Scalar<DataVector>& conformal_factor,
    const tnsr::I<DataVector, 3>& coords) {
  tnsr::I<DataVector, 3> result;
  center_of_mass_surface_integrand(make_not_null(&result), conformal_factor,
                                   coords);
  return result;
}

void center_of_mass_volume_integrand(
    gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3, Frame::Inertial>& deriv_conformal_factor,
    const tnsr::I<DataVector, 3>& coords) {
  const auto euclidean_radius = magnitude(coords);
  tenex::evaluate<ti::I>(
      result,
      3. / (4. * M_PI * pow<2>(euclidean_radius())) *
          (2. * pow<3>(conformal_factor()) * deriv_conformal_factor(ti::j) *
               coords(ti::I) * coords(ti::J) +
           pow<4>(conformal_factor()) * coords(ti::I)));
}

tnsr::I<DataVector, 3> center_of_mass_volume_integrand(
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3, Frame::Inertial>& deriv_conformal_factor,
    const tnsr::I<DataVector, 3>& coords) {
  tnsr::I<DataVector, 3> result;
  center_of_mass_volume_integrand(make_not_null(&result), conformal_factor,
                                  deriv_conformal_factor, coords);
  return result;
}

}  // namespace Xcts
