// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace gr::Solutions {

template <typename DataType>
Scalar<DataType> kerr_horizon_radius(
    const std::array<DataType, 2>& theta_phi, const double mass,
    const std::array<double, 3>& dimensionless_spin) {
  return kerr_schild_radius_from_boyer_lindquist(
      mass * (1.0 + sqrt(1.0 - square(magnitude(dimensionless_spin)))),
      theta_phi, mass, dimensionless_spin);
}

template Scalar<DataVector> kerr_horizon_radius(
    const std::array<DataVector, 2>& theta_phi, const double mass,
    const std::array<double, 3>& dimensionless_spin);

template Scalar<double> kerr_horizon_radius(
    const std::array<double, 2>& theta_phi, const double mass,
    const std::array<double, 3>& dimensionless_spin);

template <typename DataType>
Scalar<DataType> kerr_schild_radius_from_boyer_lindquist(
    const double boyer_lindquist_radius,
    const std::array<DataType, 2>& theta_phi, const double mass,
    const std::array<double, 3>& dimensionless_spin) {
  const double spin_magnitude_squared = square(magnitude(dimensionless_spin));
  const double mass_squared = square(mass);

  const auto& theta = theta_phi[0];
  const auto& phi = theta_phi[1];
  const DataType sin_theta = sin(theta);
  const DataType cos_theta = cos(theta);
  const DataType sin_phi = sin(phi);
  const DataType cos_phi = cos(phi);
  const DataType spin_dot_unit = dimensionless_spin[0] * sin_theta * cos_phi +
                                 dimensionless_spin[1] * sin_theta * sin_phi +
                                 dimensionless_spin[2] * cos_theta;

  return Scalar<DataType>{boyer_lindquist_radius *
                          sqrt(square(boyer_lindquist_radius) +
                               mass_squared * spin_magnitude_squared) /
                          sqrt(square(boyer_lindquist_radius) +
                               mass_squared * square(spin_dot_unit))};
}

template Scalar<DataVector> kerr_schild_radius_from_boyer_lindquist(
    const double boyer_lindquist_radius,
    const std::array<DataVector, 2>& theta_phi, const double mass,
    const std::array<double, 3>& dimensionless_spin);

template Scalar<double> kerr_schild_radius_from_boyer_lindquist(
    const double boyer_lindquist_radius, const std::array<double, 2>& theta_phi,
    const double mass, const std::array<double, 3>& dimensionless_spin);

}  // namespace gr::Solutions
