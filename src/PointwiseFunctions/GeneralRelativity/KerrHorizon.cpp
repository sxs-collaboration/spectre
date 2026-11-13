// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/KerrHorizon.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
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
    const std::array<double, 3>& dimensionless_spin,
    const std::array<double, 3>& boost_velocity) {
  const double spin_magnitude_squared = square(magnitude(dimensionless_spin));
  const double mass_squared = square(mass);
  const double boost_speed = magnitude(boost_velocity);
  const double one_minus_speed_squared = 1.0 - square(boost_speed);
  if (one_minus_speed_squared < 0.0 or
      equal_within_roundoff(one_minus_speed_squared, 0.0)) {
    ERROR(
        "The boost speed is either larger or too close to the speed of light.");
  }
  if (boyer_lindquist_radius < 0.0 or
      equal_within_roundoff(boyer_lindquist_radius, 0.0)) {
    ERROR("Boyer-Lindquist radius is negative or too close to zero.");
  }

  const double gamma = (equal_within_roundoff(boost_speed, 0.0)
                            ? 1.0
                            : 1.0 / sqrt(one_minus_speed_squared));

  const auto& theta = theta_phi[0];
  const auto& phi = theta_phi[1];

  // Construct unit vector
  tnsr::I<DataType, 3> boosted_unit_vector =
      make_with_value<tnsr::I<DataType, 3>>(theta, 0.0);
  boosted_unit_vector.get(0) = sin(theta) * cos(phi);
  boosted_unit_vector.get(1) = sin(theta) * sin(phi);
  boosted_unit_vector.get(2) = cos(theta);

  // Construct unit velocity vector
  // Make sure the magnitude of the velocity is not zero
  tnsr::I<DataType, 3> velocity_unit_vector =
      make_with_value<tnsr::I<DataType, 3>>(theta, 0.0);
  if (equal_within_roundoff(boost_speed, 0.0)) {
    // If the velocity is too small pick a unit vector in the y direction
    velocity_unit_vector.get(1) = 1.0;
  } else {
    for (size_t i = 0; i < 3; i++) {
      velocity_unit_vector.get(i) = gsl::at(boost_velocity, i) / boost_speed;
    }
  }

  // Compute projection along the velocity direction
  Scalar<DataType> boosted_unit_vector_dot_velocity_unit_vector =
      make_with_value<Scalar<DataType>>(theta, 0.0);
  get(boosted_unit_vector_dot_velocity_unit_vector) =
      boosted_unit_vector.get(0) * velocity_unit_vector.get(0);
  for (size_t i = 1; i < 3; i++) {
    get(boosted_unit_vector_dot_velocity_unit_vector) +=
        boosted_unit_vector.get(i) * velocity_unit_vector.get(i);
  }

  // Obtain the normal vector in the rest frame by by reverting the Lorentz
  // contraction along the direction along the velocity
  // We first decompose in parallel and orthogonal directions, then multiply the
  // parallel component by the Lorentz factor
  tnsr::I<DataType, 3> boosted_unit_vector_parallel{};
  tnsr::I<DataType, 3> boosted_unit_vector_orthogonal{};
  tenex::evaluate<ti::I>(make_not_null(&boosted_unit_vector_parallel),
                         boosted_unit_vector_dot_velocity_unit_vector() *
                             velocity_unit_vector(ti::I));
  tenex::evaluate<ti::I>(
      make_not_null(&boosted_unit_vector_orthogonal),
      boosted_unit_vector(ti::I) - boosted_unit_vector_parallel(ti::I));

  tnsr::I<DataType, 3> unboosted_unit_vector{};
  tenex::evaluate<ti::I>(make_not_null(&unboosted_unit_vector),
                         gamma * boosted_unit_vector_parallel(ti::I) +
                             boosted_unit_vector_orthogonal(ti::I));

  // Normalize the transformed vector
  // This will be the direction-dependent Lorentz factor we need to apply to the
  // Kerr-Schild radius and should be equal to
  // sqrt(1.0 + (square(gamma) - 1.0) *
  //                  square(get(boosted_unit_vector_dot_velocity_unit_vector)))
  const Scalar<DataType> magnitude_unboosted_unit_vector =
      magnitude(unboosted_unit_vector);

  for (size_t i = 0; i < 3; i++) {
    unboosted_unit_vector.get(i) /= get(magnitude_unboosted_unit_vector);
  }

  // To apply the Kerr-Schild-Boyer-Lindquist relation in the rest frame, we
  // need to project the spin using the rest frame unit vector
  Scalar<DataType> spin_dot_unit =
      make_with_value<Scalar<DataType>>(theta, 0.0);
  get(spin_dot_unit) =
      gsl::at(dimensionless_spin, 0) * unboosted_unit_vector.get(0);
  for (size_t i = 1; i < 3; i++) {
    get(spin_dot_unit) +=
        gsl::at(dimensionless_spin, i) * unboosted_unit_vector.get(i);
  }

  // Compute the Kerr-Schild-Boyer-Lindquist relation in the boosted frame
  // This is the same as the Kerr-Schild-Boyer-Lindquist relation in the rest
  // frame, just multiplied by the direction-dependent Lorentz contraction
  // factor
  Scalar<DataType> ks_radius_in_boosted_frame =
      make_with_value<Scalar<DataType>>(theta, 0.0);
  get(ks_radius_in_boosted_frame) =
      boyer_lindquist_radius *
      sqrt(square(boyer_lindquist_radius) +
           mass_squared * spin_magnitude_squared) /
      (get(magnitude_unboosted_unit_vector) *
       sqrt(square(boyer_lindquist_radius) +
            mass_squared * square(get(spin_dot_unit))));

  return ks_radius_in_boosted_frame;
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
