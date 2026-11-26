// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/KerrHorizon.hpp"

#include <array>
#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/StdHelpers.hpp"

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
    const std::array<double, 3>& dimensionless_spin,
    const std::array<double, 3>& boost_velocity) {
  const double speed_squared = dot(boost_velocity, boost_velocity);
  if (speed_squared > 1.0 or equal_within_roundoff(speed_squared, 1.0)) {
    ERROR("The boost speed "
          << sqrt(speed_squared) << " (velocity " << boost_velocity
          << ") is not smaller than or is too close to the speed of light.");
  }
  if (boyer_lindquist_radius < 0.0 or
      equal_within_roundoff(boyer_lindquist_radius, 0.0)) {
    ERROR("The Boyer-Lindquist radius "
          << boyer_lindquist_radius
          << " is not positive or is too close to zero.");
  }

  // See the documentation in the header for the derivation
  const double lorentz_factor_squared = 1.0 / (1.0 - speed_squared);
  const double lorentz_factor = sqrt(lorentz_factor_squared);
  // (gamma - 1) / v^2, written without the division by v^2
  const double kappa = lorentz_factor_squared / (1.0 + lorentz_factor);
  const double spin_dot_velocity = dot(dimensionless_spin, boost_velocity);
  const double mass_squared = square(mass);
  const double boyer_lindquist_radius_squared = square(boyer_lindquist_radius);
  const double numerator =
      boyer_lindquist_radius *
      sqrt(boyer_lindquist_radius_squared +
           mass_squared * dot(dimensionless_spin, dimensionless_spin));

  const auto& theta = theta_phi[0];
  const auto& phi = theta_phi[1];
  const DataType normal_dot_velocity =
      sin(theta) *
          (boost_velocity[0] * cos(phi) + boost_velocity[1] * sin(phi)) +
      boost_velocity[2] * cos(theta);
  const DataType normal_dot_spin =
      sin(theta) * (dimensionless_spin[0] * cos(phi) +
                    dimensionless_spin[1] * sin(phi)) +
      dimensionless_spin[2] * cos(theta);

  return Scalar<DataType>{DataType{
      numerator /
      sqrt(boyer_lindquist_radius_squared *
               (1.0 + lorentz_factor_squared * square(normal_dot_velocity)) +
           mass_squared * square(normal_dot_spin + kappa * normal_dot_velocity *
                                                       spin_dot_velocity))}};
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
