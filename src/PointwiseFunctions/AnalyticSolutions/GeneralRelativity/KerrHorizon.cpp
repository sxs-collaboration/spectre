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
  const std::array<double, 3> boost_velocity = {0.0, 0.0, 0.0};
  return kerr_schild_radius_from_boyer_lindquist(
      mass * (1.0 + sqrt(1.0 - square(magnitude(dimensionless_spin)))),
      theta_phi, mass, dimensionless_spin, boost_velocity);
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
    const std::array<double, 3>& dimensionless_spin,
    const std::array<double, 3>& boost_velocity) {
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

  const double boost_velocity_x = boost_velocity[0];
  const double boost_velocity_y = boost_velocity[1];
  const double boost_velocity_z = boost_velocity[2];
  const double lorentz_factor_squared =
      1.0 / (1.0 - (square(boost_velocity_x) + square(boost_velocity_y) +
                    square(boost_velocity_z)));

  Scalar<DataType> unboosted_kerr_schild_radius =
      Scalar<DataType>{boyer_lindquist_radius *
                       sqrt(square(boyer_lindquist_radius) +
                            mass_squared * spin_magnitude_squared) /
                       sqrt(square(boyer_lindquist_radius) +
                            mass_squared * square(spin_dot_unit))};

  return Scalar<DataType>{sqrt(square(get(unboosted_kerr_schild_radius)) +
                          lorentz_factor_squared *
                              square(get(unboosted_kerr_schild_radius)) *
                              square(cos_phi * sin_theta * boost_velocity_x +
                                     sin_phi * sin_theta * boost_velocity_y +
                                     cos_theta * boost_velocity_z))};
}

template Scalar<DataVector> kerr_schild_radius_from_boyer_lindquist(
    const double boyer_lindquist_radius,
    const std::array<DataVector, 2>& theta_phi, const double mass,
    const std::array<double, 3>& dimensionless_spin,
    const std::array<double, 3>& boost_velocity);

template Scalar<double> kerr_schild_radius_from_boyer_lindquist(
    const double boyer_lindquist_radius, const std::array<double, 2>& theta_phi,
    const double mass, const std::array<double, 3>& dimensionless_spin,
    const std::array<double, 3>& boost_velocity);

}  // namespace gr::Solutions
