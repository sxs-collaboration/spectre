// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/SphericalToCartesianPfaffian.hpp"

#include <cmath>
#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain::CoordinateMaps {
SphericalToCartesianPfaffian::SphericalToCartesianPfaffian() = default;
SphericalToCartesianPfaffian::SphericalToCartesianPfaffian(
    SphericalToCartesianPfaffian&&) = default;
SphericalToCartesianPfaffian::SphericalToCartesianPfaffian(
    const SphericalToCartesianPfaffian&) = default;
SphericalToCartesianPfaffian& SphericalToCartesianPfaffian::operator=(
    const SphericalToCartesianPfaffian&) = default;
SphericalToCartesianPfaffian& SphericalToCartesianPfaffian::operator=(
    SphericalToCartesianPfaffian&&) = default;

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3>
SphericalToCartesianPfaffian::operator()(
    const std::array<T, 3>& source_coords) const {
  const auto& [r, theta, phi] = source_coords;
  return {
      {r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)}};
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
std::optional<std::array<double, 3>> SphericalToCartesianPfaffian::inverse(
    const std::array<double, 3>& target_coords) const {
  const auto& [x, y, z] = target_coords;
  if (UNLIKELY(y == 0.0 and x == 0.0)) {
    if (UNLIKELY(z == 0.0)) {
      return std::array{0.0, 0.5 * M_PI, 0.0};
    } else {
      return std::array{std::abs(z), z > 0.0 ? 0.0 : M_PI, 0.0};
    }
  } else {
    const double r = std::hypot(x, y, z);
    const double phi = atan2(y, x);
    return std::array{r, acos(z / r), phi < 0.0 ? phi + 2.0 * M_PI : phi};
  }
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
SphericalToCartesianPfaffian::jacobian(
    const std::array<T, 3>& source_coords) const {
  const auto& [r, theta, phi] = source_coords;
  using DataType = tt::remove_cvref_wrap_t<T>;
  tnsr::Ij<DataType, 3, Frame::NoFrame> jacobian_matrix{
      make_with_value<DataType>(dereference_wrapper(r), 0.0)};
  // Pfaffian basis means phi components are 1 / sin_theta times those of a
  // coord basis
  const auto& cos_theta = get<2, 0>(jacobian_matrix) = cos(theta);
  const auto& sin_theta = get<2, 1>(jacobian_matrix) = sin(theta);
  const auto& cos_phi = get<1, 2>(jacobian_matrix) = cos(phi);
  const auto& sin_phi = get<0, 2>(jacobian_matrix) = sin(phi);
  get<0, 0>(jacobian_matrix) = sin_theta * cos_phi;
  get<1, 0>(jacobian_matrix) = sin_theta * sin_phi;
  get<0, 1>(jacobian_matrix) = r * cos_theta * cos_phi;
  get<1, 1>(jacobian_matrix) = r * cos_theta * sin_phi;
  get<2, 1>(jacobian_matrix) *= -r;
  get<0, 2>(jacobian_matrix) *= -r;
  get<1, 2>(jacobian_matrix) *= r;
  // get<2, 2>(jacobian_matrix) is zero
  return jacobian_matrix;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
SphericalToCartesianPfaffian::inv_jacobian(
    const std::array<T, 3>& source_coords) const {
  const auto& [r, theta, phi] = source_coords;
  using DataType = tt::remove_cvref_wrap_t<T>;
  tnsr::Ij<DataType, 3, Frame::NoFrame> inv_jacobian_matrix{
      make_with_value<DataType>(dereference_wrapper(r), 0.0)};
  // Pfaffian basis means phi components are sin_theta times those of a coord
  // basis
  const auto& cos_theta = get<0, 2>(inv_jacobian_matrix) = cos(theta);
  const auto& sin_theta = get<1, 2>(inv_jacobian_matrix) = sin(theta);
  const auto& cos_phi = get<2, 1>(inv_jacobian_matrix) = cos(phi);
  const auto& sin_phi = get<2, 0>(inv_jacobian_matrix) = sin(phi);
  const auto& one_over_r = get<2, 2>(inv_jacobian_matrix) = 1.0/r;
  get<0, 0>(inv_jacobian_matrix) = sin_theta * cos_phi;
  get<0, 1>(inv_jacobian_matrix) = sin_theta * sin_phi;
  get<1, 0>(inv_jacobian_matrix) = cos_theta * cos_phi * one_over_r;
  get<1, 1>(inv_jacobian_matrix) = cos_theta * sin_phi * one_over_r;
  get<1, 2>(inv_jacobian_matrix) *= -one_over_r;
  get<2, 0>(inv_jacobian_matrix) *= -one_over_r;
  get<2, 1>(inv_jacobian_matrix) *= one_over_r;
  get<2, 2>(inv_jacobian_matrix) *= 0.0;
  return inv_jacobian_matrix;
}

void SphericalToCartesianPfaffian::pup(PUP::er& /*p*/) {}

bool operator==(const SphericalToCartesianPfaffian& /*lhs*/,
                const SphericalToCartesianPfaffian& /*rhs*/) {
  return true;
}

bool operator!=(const SphericalToCartesianPfaffian& lhs,
                const SphericalToCartesianPfaffian& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE_DTYPE(_, data)                                           \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>               \
  SphericalToCartesianPfaffian::operator()(                                  \
      const std::array<DTYPE(data), 3>& source_coords) const;                \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  SphericalToCartesianPfaffian::jacobian(                                    \
      const std::array<DTYPE(data), 3>& source_coords) const;                \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  SphericalToCartesianPfaffian::inv_jacobian(                                \
      const std::array<DTYPE(data), 3>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_DTYPE,
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))
}  // namespace domain::CoordinateMaps
