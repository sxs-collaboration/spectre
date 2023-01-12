// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/ComovingMagneticFieldMagnitude.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean::Tags {

void ComovingMagneticFieldMagnitudeCompute::function(
    const gsl::not_null<Scalar<DataVector>*> comoving_magnetic_field_magnitude,
    const tnsr::I<DataVector, 3>& magnetic_field,
    const tnsr::I<DataVector, 3>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric) {
  Variables<tmpl::list<hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>,
                       hydro::Tags::MagneticFieldSquared<DataVector>>>
      temp_tensors{get(lorentz_factor).size()};

  const auto& magnetic_field_dot_spatial_velocity =
      get<hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>>(
          temp_tensors);
  dot_product(
      make_not_null(
          &get<hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>>(
              temp_tensors)),
      magnetic_field, spatial_velocity, spatial_metric);

  const auto& magnetic_field_squared =
      get<hydro::Tags::MagneticFieldSquared<DataVector>>(temp_tensors);
  dot_product(make_not_null(&get<hydro::Tags::MagneticFieldSquared<DataVector>>(
                  temp_tensors)),
              magnetic_field, magnetic_field, spatial_metric);

  get(*comoving_magnetic_field_magnitude) =
      sqrt(get(magnetic_field_squared) / square(get(lorentz_factor)) +
           square(get(magnetic_field_dot_spatial_velocity)));
}

}  // namespace grmhd::ValenciaDivClean::Tags
