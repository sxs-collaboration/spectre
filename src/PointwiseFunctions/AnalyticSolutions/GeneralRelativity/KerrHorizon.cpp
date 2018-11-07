// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"                 // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp" // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace gr {
namespace Solutions {

template <typename DataType>
Scalar<DataType> kerr_horizon_radius(
    const std::array<DataType, 2>& theta_phi, const double& mass,
    const std::array<double, 3>& dimensionless_spin) noexcept {
  const double spin_magnitude_squared = square(magnitude(dimensionless_spin));
  const double mass_squared = square(mass);

  const double equatorial_radius_squared =
      2.0 * mass_squared * (1.0 + sqrt(1.0 - spin_magnitude_squared));
  const double polar_radius_squared =
      mass_squared * square(1.0 + sqrt(1.0 - spin_magnitude_squared));

  const auto& theta = theta_phi[0];
  const auto& phi = theta_phi[1];
  const DataType sin_theta = sin(theta);
  const DataType cos_theta = cos(theta);
  const DataType sin_phi = sin(phi);
  const DataType cos_phi = cos(phi);

  auto denominator =
      make_with_value<DataType>(theta_phi[0], polar_radius_squared);
  denominator += mass_squared * dimensionless_spin[0] * dimensionless_spin[0] *
                 square(sin_theta * cos_phi);
  denominator += mass_squared * dimensionless_spin[1] * dimensionless_spin[1] *
                 square(sin_theta * sin_phi);
  denominator += mass_squared * dimensionless_spin[2] * dimensionless_spin[2] *
                 square(cos_theta);
  denominator += 2.0 * mass_squared * dimensionless_spin[0] *
                 dimensionless_spin[1] * square(sin_theta) * sin_phi * cos_phi;
  denominator += 2.0 * mass_squared * dimensionless_spin[0] *
                 dimensionless_spin[2] * sin_theta * cos_theta * cos_phi;
  denominator += 2.0 * mass_squared * dimensionless_spin[1] *
                 dimensionless_spin[2] * sin_theta * cos_theta * sin_phi;

  auto radius_squared =
      make_with_value<DataType>(theta_phi[0], polar_radius_squared);
  radius_squared *= equatorial_radius_squared;
  radius_squared /= denominator;

  return Scalar<DataType>{sqrt(radius_squared)};
}

template Scalar<DataVector> kerr_horizon_radius(
    const std::array<DataVector, 2>& theta_phi, const double& mass,
    const std::array<double, 3>& dimensionless_spin) noexcept;

template Scalar<double> kerr_horizon_radius(
    const std::array<double, 2>& theta_phi, const double& mass,
    const std::array<double, 3>& dimensionless_spin) noexcept;

}  // namespace Solutions
}  // namespace gr

