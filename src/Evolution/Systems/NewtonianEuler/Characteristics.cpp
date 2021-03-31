// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Characteristics.hpp"

#include <cmath>
#include <iterator>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace NewtonianEuler {

template <size_t Dim>
void characteristic_speeds(
    const gsl::not_null<std::array<DataVector, Dim + 2>*> char_speeds,
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& sound_speed,
    const tnsr::i<DataVector, Dim>& normal) noexcept {
  auto& characteristic_speeds = *char_speeds;
  characteristic_speeds =
      make_array<Dim + 2>(DataVector(get(dot_product(velocity, normal))));

  characteristic_speeds[0] -= get(sound_speed);
  characteristic_speeds[Dim + 1] += get(sound_speed);
}

template <size_t Dim>
std::array<DataVector, Dim + 2> characteristic_speeds(
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& sound_speed,
    const tnsr::i<DataVector, Dim>& normal) noexcept {
  std::array<DataVector, Dim + 2> char_speeds{};
  characteristic_speeds(make_not_null(&char_speeds), velocity, sound_speed,
                        normal);
  return char_speeds;
}

template <>
Matrix right_eigenvectors<1>(const tnsr::I<double, 1>& velocity,
                             const Scalar<double>& sound_speed_squared,
                             const Scalar<double>& specific_enthalpy,
                             const Scalar<double>& kappa_over_density,
                             const tnsr::i<double, 1>& unit_normal) noexcept {
  ASSERT(equal_within_roundoff(get(magnitude(unit_normal)), 1.),
         "Expected unit normal, but got normal with magnitude "
             << get(magnitude(unit_normal)));

  const double u = get<0>(velocity);
  const double n_x = get<0>(unit_normal);
  const double c = sqrt(get(sound_speed_squared));
  const double v_n = get(dot_product(velocity, unit_normal));

  Matrix result(3, 3);
  result(0, 0) = 1.;
  result(0, 1) = get(kappa_over_density);
  result(0, 2) = 1.;
  result(1, 0) = u - n_x * c;
  result(1, 1) = get(kappa_over_density) * u;
  result(1, 2) = u + n_x * c;
  result(2, 0) = get(specific_enthalpy) - c * v_n;
  result(2, 1) = get(kappa_over_density) * get(specific_enthalpy) -
                 get(sound_speed_squared);
  result(2, 2) = get(specific_enthalpy) + c * v_n;
  return result;
}

template <>
Matrix right_eigenvectors<2>(const tnsr::I<double, 2>& velocity,
                             const Scalar<double>& sound_speed_squared,
                             const Scalar<double>& specific_enthalpy,
                             const Scalar<double>& kappa_over_density,
                             const tnsr::i<double, 2>& unit_normal) noexcept {
  ASSERT(equal_within_roundoff(get(magnitude(unit_normal)), 1.),
         "Expected unit normal, but got normal with magnitude "
             << get(magnitude(unit_normal)));

  const double u = get<0>(velocity);
  const double v = get<1>(velocity);
  const double n_x = get<0>(unit_normal);
  const double n_y = get<1>(unit_normal);
  const double c = sqrt(get(sound_speed_squared));
  const double v_n = get(dot_product(velocity, unit_normal));

  Matrix result(4, 4);
  result(0, 0) = 1.;
  result(0, 1) = get(kappa_over_density);
  result(0, 2) = 0.;
  result(0, 3) = 1.;
  result(1, 0) = u - n_x * c;
  result(1, 1) = get(kappa_over_density) * u;
  result(1, 2) = -n_y;
  result(1, 3) = u + n_x * c;
  result(2, 0) = v - n_y * c;
  result(2, 1) = get(kappa_over_density) * v;
  result(2, 2) = n_x;
  result(2, 3) = v + n_y * c;
  result(3, 0) = get(specific_enthalpy) - c * v_n;
  result(3, 1) = get(kappa_over_density) * get(specific_enthalpy) -
                 get(sound_speed_squared);
  result(3, 2) = -n_y * u + n_x * v;
  result(3, 3) = get(specific_enthalpy) + c * v_n;
  return result;
}

template <>
Matrix right_eigenvectors<3>(const tnsr::I<double, 3>& velocity,
                             const Scalar<double>& sound_speed_squared,
                             const Scalar<double>& specific_enthalpy,
                             const Scalar<double>& kappa_over_density,
                             const tnsr::i<double, 3>& unit_normal) noexcept {
  ASSERT(equal_within_roundoff(get(magnitude(unit_normal)), 1.),
         "Expected unit normal, but got normal with magnitude "
             << get(magnitude(unit_normal)));

  const double u = get<0>(velocity);
  const double v = get<1>(velocity);
  const double w = get<2>(velocity);
  const double n_x = get<0>(unit_normal);
  const double n_y = get<1>(unit_normal);
  const double n_z = get<2>(unit_normal);
  const double c = sqrt(get(sound_speed_squared));
  const double v_n = get(dot_product(velocity, unit_normal));

  Matrix result(5, 5);
  result(0, 0) = 1.;
  result(1, 0) = u - n_x * c;
  result(2, 0) = v - n_y * c;
  result(3, 0) = w - n_z * c;
  result(4, 0) = get(specific_enthalpy) - v_n * c;
  result(0, 1) = get(kappa_over_density);
  result(1, 1) = get(kappa_over_density) * u;
  result(2, 1) = get(kappa_over_density) * v;
  result(3, 1) = get(kappa_over_density) * w;
  result(4, 1) = get(kappa_over_density) * get(specific_enthalpy) -
                 get(sound_speed_squared);
  result(0, 4) = 1.;
  result(1, 4) = u + n_x * c;
  result(2, 4) = v + n_y * c;
  result(3, 4) = w + n_z * c;
  result(4, 4) = get(specific_enthalpy) + v_n * c;

  // There is some degeneracy in the choice of the column eigenvectors. The row
  // eigenvectors are chosen so that the largest of (n_x, n_y, n_z) appears in
  // the denominator, because this avoids division by zero. Here we make the
  // consistent choice of right eigenvectors.
  const auto index_of_largest =
      std::distance(unit_normal.begin(), alg::max_element(unit_normal, [
                    ](const double n1, const double n2) noexcept {
                      return fabs(n1) < fabs(n2);
                    }));
  if (index_of_largest == 0) {
    // right eigenvectors corresponding to left eigenvectors with 1/n_x terms
    result(0, 2) = 0.;
    result(1, 2) = -n_y;
    result(2, 2) = n_x;
    result(3, 2) = 0.;
    result(4, 2) = -n_y * u + n_x * v;
    result(0, 3) = 0.;
    result(1, 3) = -n_z;
    result(2, 3) = 0.;
    result(3, 3) = n_x;
    result(4, 3) = -n_z * u + n_x * w;
  } else if (index_of_largest == 1) {
    // right eigenvectors corresponding to left eigenvectors with 1/n_y terms
    result(0, 2) = 0.;
    result(1, 2) = -n_y;
    result(2, 2) = n_x;
    result(3, 2) = 0.;
    result(4, 2) = -n_y * u + n_x * v;
    result(0, 3) = 0.;
    result(1, 3) = 0.;
    result(2, 3) = -n_z;
    result(3, 3) = n_y;
    result(4, 3) = -n_z * v + n_y * w;
  } else {
    // right eigenvectors corresponding to left eigenvectors with 1/n_z terms
    result(0, 2) = 0.;
    result(1, 2) = -n_z;
    result(2, 2) = 0.;
    result(3, 2) = n_x;
    result(4, 2) = -n_z * u + n_x * w;
    result(0, 3) = 0.;
    result(1, 3) = 0.;
    result(2, 3) = -n_z;
    result(3, 3) = n_y;
    result(4, 3) = -n_z * v + n_y * w;
  }
  return result;
}

template <>
Matrix left_eigenvectors<1>(const tnsr::I<double, 1>& velocity,
                            const Scalar<double>& sound_speed_squared,
                            const Scalar<double>& specific_enthalpy,
                            const Scalar<double>& kappa_over_density,
                            const tnsr::i<double, 1>& unit_normal) noexcept {
  ASSERT(equal_within_roundoff(get(magnitude(unit_normal)), 1.),
         "Expected unit normal, but got normal with magnitude "
             << get(magnitude(unit_normal)));

  const double velocity_squared = get(dot_product(velocity, velocity));
  const double u = get<0>(velocity);
  const double n_x = get<0>(unit_normal);
  const double c = sqrt(get(sound_speed_squared));
  const double v_n = get(dot_product(velocity, unit_normal));

  // Temporary with a useful combination, as per Kulikovskii Ch3
  const double b_times_theta =
      get(kappa_over_density) * (velocity_squared - get(specific_enthalpy)) +
      get(sound_speed_squared);

  Matrix result(3, 3);
  result(0, 0) = 0.5 * (b_times_theta + c * v_n) / get(sound_speed_squared);
  result(0, 1) =
      -0.5 * (get(kappa_over_density) * u + n_x * c) / get(sound_speed_squared);
  result(0, 2) = 0.5 * get(kappa_over_density) / get(sound_speed_squared);
  result(1, 0) =
      (get(specific_enthalpy) - velocity_squared) / get(sound_speed_squared);
  result(1, 1) = u / get(sound_speed_squared);
  result(1, 2) = -1. / get(sound_speed_squared);
  result(2, 0) = 0.5 * (b_times_theta - c * v_n) / get(sound_speed_squared);
  result(2, 1) =
      -0.5 * (get(kappa_over_density) * u - n_x * c) / get(sound_speed_squared);
  result(2, 2) = 0.5 * get(kappa_over_density) / get(sound_speed_squared);
  return result;
}

template <>
Matrix left_eigenvectors<2>(const tnsr::I<double, 2>& velocity,
                            const Scalar<double>& sound_speed_squared,
                            const Scalar<double>& specific_enthalpy,
                            const Scalar<double>& kappa_over_density,
                            const tnsr::i<double, 2>& unit_normal) noexcept {
  ASSERT(equal_within_roundoff(get(magnitude(unit_normal)), 1.),
         "Expected unit normal, but got normal with magnitude "
             << get(magnitude(unit_normal)));

  const double velocity_squared = get(dot_product(velocity, velocity));
  const double u = get<0>(velocity);
  const double v = get<1>(velocity);
  const double n_x = get<0>(unit_normal);
  const double n_y = get<1>(unit_normal);
  const double c = sqrt(get(sound_speed_squared));
  const double v_n = get(dot_product(velocity, unit_normal));

  // Temporary with a useful combination, as per Kulikovskii Ch3
  const double b_times_theta =
      get(kappa_over_density) * (velocity_squared - get(specific_enthalpy)) +
      get(sound_speed_squared);

  Matrix result(4, 4);
  result(0, 0) = 0.5 * (b_times_theta + c * v_n) / get(sound_speed_squared);
  result(0, 1) =
      -0.5 * (get(kappa_over_density) * u + n_x * c) / get(sound_speed_squared);
  result(0, 2) =
      -0.5 * (get(kappa_over_density) * v + n_y * c) / get(sound_speed_squared);
  result(0, 3) = 0.5 * get(kappa_over_density) / get(sound_speed_squared);
  result(1, 0) =
      (get(specific_enthalpy) - velocity_squared) / get(sound_speed_squared);
  result(1, 1) = u / get(sound_speed_squared);
  result(1, 2) = v / get(sound_speed_squared);
  result(1, 3) = -1. / get(sound_speed_squared);
  result(2, 0) = n_y * u - n_x * v;
  result(2, 1) = -n_y;
  result(2, 2) = n_x;
  result(2, 3) = 0.;
  result(3, 0) = 0.5 * (b_times_theta - c * v_n) / get(sound_speed_squared);
  result(3, 1) =
      -0.5 * (get(kappa_over_density) * u - n_x * c) / get(sound_speed_squared);
  result(3, 2) =
      -0.5 * (get(kappa_over_density) * v - n_y * c) / get(sound_speed_squared);
  result(3, 3) = 0.5 * get(kappa_over_density) / get(sound_speed_squared);
  return result;
}

template <>
Matrix left_eigenvectors<3>(const tnsr::I<double, 3>& velocity,
                            const Scalar<double>& sound_speed_squared,
                            const Scalar<double>& specific_enthalpy,
                            const Scalar<double>& kappa_over_density,
                            const tnsr::i<double, 3>& unit_normal) noexcept {
  ASSERT(equal_within_roundoff(get(magnitude(unit_normal)), 1.),
         "Expected unit normal, but got normal with magnitude "
             << get(magnitude(unit_normal)));

  const double velocity_squared = get(dot_product(velocity, velocity));
  const double u = get<0>(velocity);
  const double v = get<1>(velocity);
  const double w = get<2>(velocity);
  const double n_x = get<0>(unit_normal);
  const double n_y = get<1>(unit_normal);
  const double n_z = get<2>(unit_normal);
  const double c = sqrt(get(sound_speed_squared));
  const double v_n = get(dot_product(velocity, unit_normal));

  // Temporary with a useful combination, as per Kulikovskii Ch3
  const double b_times_theta =
      get(kappa_over_density) * (velocity_squared - get(specific_enthalpy)) +
      get(sound_speed_squared);

  Matrix result(5, 5);
  result(0, 0) = 0.5 * (b_times_theta + c * v_n) / get(sound_speed_squared);
  result(0, 1) =
      -0.5 * (get(kappa_over_density) * u + n_x * c) / get(sound_speed_squared);
  result(0, 2) =
      -0.5 * (get(kappa_over_density) * v + n_y * c) / get(sound_speed_squared);
  result(0, 3) =
      -0.5 * (get(kappa_over_density) * w + n_z * c) / get(sound_speed_squared);
  result(0, 4) = 0.5 * get(kappa_over_density) / get(sound_speed_squared);
  result(1, 0) =
      (get(specific_enthalpy) - velocity_squared) / get(sound_speed_squared);
  result(1, 1) = u / get(sound_speed_squared);
  result(1, 2) = v / get(sound_speed_squared);
  result(1, 3) = w / get(sound_speed_squared);
  result(1, 4) = -1. / get(sound_speed_squared);
  result(4, 0) = 0.5 * (b_times_theta - c * v_n) / get(sound_speed_squared);
  result(4, 1) =
      -0.5 * (get(kappa_over_density) * u - n_x * c) / get(sound_speed_squared);
  result(4, 2) =
      -0.5 * (get(kappa_over_density) * v - n_y * c) / get(sound_speed_squared);
  result(4, 3) =
      -0.5 * (get(kappa_over_density) * w - n_z * c) / get(sound_speed_squared);
  result(4, 4) = 0.5 * get(kappa_over_density) / get(sound_speed_squared);

  // There is some degeneracy in the choice of the row eigenvectors. Here, we
  // use rows where the largest of (n_x, n_y, n_z) appears in the denominator,
  // because this avoids division by zero. A consistent choice of right
  // eigenvectors must be made.
  const auto index_of_largest =
      std::distance(unit_normal.begin(), alg::max_element(unit_normal, [
                    ](const double n1, const double n2) noexcept {
                      return fabs(n1) < fabs(n2);
                    }));
  if (index_of_largest == 0) {
    // left eigenvectors with 1/n_x terms
    result(2, 0) = (n_y * v_n - v) / n_x;
    result(2, 1) = -n_y;
    result(2, 2) = (1. - square(n_y)) / n_x;
    result(2, 3) = -n_y * n_z / n_x;
    result(2, 4) = 0.;
    result(3, 0) = (n_z * v_n - w) / n_x;
    result(3, 1) = -n_z;
    result(3, 2) = -n_y * n_z / n_x;
    result(3, 3) = n_x + square(n_y) / n_x;
    result(3, 4) = 0.;
  } else if (index_of_largest == 1) {
    // left eigenvectors with 1/n_y terms
    result(2, 0) = (u - n_x * v_n) / n_y;
    result(2, 1) = (-1. + square(n_x)) / n_y;
    result(2, 2) = n_x;
    result(2, 3) = n_x * n_z / n_y;
    result(2, 4) = 0.;
    result(3, 0) = (n_z * v_n - w) / n_y;
    result(3, 1) = -n_x * n_z / n_y;
    result(3, 2) = -n_z;
    result(3, 3) = n_y + square(n_x) / n_y;
    result(3, 4) = 0.;
  } else {
    // left eigenvectors with 1/n_z terms
    result(2, 0) = (u - n_x * v_n) / n_z;
    result(2, 1) = (-1. + square(n_x)) / n_z;
    result(2, 2) = n_x * n_y / n_z;
    result(2, 3) = n_x;
    result(2, 4) = 0.;
    result(3, 0) = (v - n_y * v_n) / n_z;
    result(3, 1) = n_x * n_y / n_z;
    result(3, 2) = (-1. + square(n_y)) / n_z;
    result(3, 3) = n_y;
    result(3, 4) = 0.;
  }
  return result;
}

}  // namespace NewtonianEuler

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void NewtonianEuler::characteristic_speeds(                         \
      const gsl::not_null<std::array<DataVector, DIM(data) + 2>*> char_speeds, \
      const tnsr::I<DataVector, DIM(data)>& velocity,                          \
      const Scalar<DataVector>& sound_speed,                                   \
      const tnsr::i<DataVector, DIM(data)>& normal) noexcept;                  \
  template std::array<DataVector, DIM(data) + 2>                               \
  NewtonianEuler::characteristic_speeds(                                       \
      const tnsr::I<DataVector, DIM(data)>& velocity,                          \
      const Scalar<DataVector>& sound_speed,                                   \
      const tnsr::i<DataVector, DIM(data)>& normal) noexcept;                  \
  template struct NewtonianEuler::Tags::ComputeLargestCharacteristicSpeed<DIM( \
      data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond
