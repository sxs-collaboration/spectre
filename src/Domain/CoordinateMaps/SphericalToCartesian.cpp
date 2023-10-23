// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/SphericalToCartesian.hpp"

#include <cmath>
#include <cstddef>
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain::CoordinateMaps {

template <size_t Dim>
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, Dim>
SphericalToCartesian<Dim>::operator()(
    const std::array<T, Dim>& spherical_coords) const {
  if constexpr (Dim == 2) {
    const auto& r = spherical_coords[radial_coord];
    const auto& phi = spherical_coords[polar_coord];
    return {{r * cos(phi), r * sin(phi)}};
  } else {
    const auto& r = spherical_coords[radial_coord];
    const auto& theta = spherical_coords[polar_coord];
    const auto& phi = spherical_coords[azimuth_coord];
    return {
        {r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)}};
  }
}

template <size_t Dim>
std::optional<std::array<double, Dim>> SphericalToCartesian<Dim>::inverse(
    const std::array<double, Dim>& cartesian_coords) const {
  std::array<double, Dim> spherical_coords{};
  if constexpr (Dim == 2) {
    const auto& [x, y] = cartesian_coords;
    auto& r = spherical_coords[radial_coord];
    auto& phi = spherical_coords[polar_coord];
    r = sqrt(square(x) + square(y));
    phi = atan2(y, x);
    if (phi < 0.) {
      phi += 2. * M_PI;
    }
  } else {
    const auto& [x, y, z] = cartesian_coords;
    auto& r = spherical_coords[radial_coord];
    auto& theta = spherical_coords[polar_coord];
    auto& phi = spherical_coords[azimuth_coord];
    r = sqrt(square(x) + square(y) + square(z));
    theta = acos(z / r);
    phi = atan2(y, x);
    if (phi < 0.) {
      phi += 2. * M_PI;
    }
  }
  return {std::move(spherical_coords)};
}

template <size_t Dim>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame>
SphericalToCartesian<Dim>::jacobian(
    const std::array<T, Dim>& spherical_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  auto jacobian_matrix =
      make_with_value<tnsr::Ij<ReturnType, Dim, Frame::NoFrame>>(
          spherical_coords[0], 0.);
  if constexpr (Dim == 2) {
    const auto& r = spherical_coords[radial_coord];
    const auto& phi = spherical_coords[polar_coord];
    get<0, radial_coord>(jacobian_matrix) = cos(phi);
    get<1, radial_coord>(jacobian_matrix) = sin(phi);
    get<0, polar_coord>(jacobian_matrix) = -r * sin(phi);
    get<1, polar_coord>(jacobian_matrix) = r * cos(phi);
  } else {
    const auto& r = spherical_coords[radial_coord];
    const auto& theta = spherical_coords[polar_coord];
    const auto& phi = spherical_coords[azimuth_coord];
    get<0, radial_coord>(jacobian_matrix) = sin(theta) * cos(phi);
    get<1, radial_coord>(jacobian_matrix) = sin(theta) * sin(phi);
    get<2, radial_coord>(jacobian_matrix) = cos(theta);
    get<0, polar_coord>(jacobian_matrix) = r * cos(theta) * cos(phi);
    get<1, polar_coord>(jacobian_matrix) = r * cos(theta) * sin(phi);
    get<2, polar_coord>(jacobian_matrix) = -r * sin(theta);
    get<0, azimuth_coord>(jacobian_matrix) = -r * sin(theta) * sin(phi);
    get<1, azimuth_coord>(jacobian_matrix) = r * sin(theta) * cos(phi);
    get<2, azimuth_coord>(jacobian_matrix) = 0.;
  }
  return jacobian_matrix;
}

template <size_t Dim>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame>
SphericalToCartesian<Dim>::inv_jacobian(
    const std::array<T, Dim>& spherical_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  auto inv_jacobian_matrix =
      make_with_value<tnsr::Ij<ReturnType, Dim, Frame::NoFrame>>(
          spherical_coords[0], 0.);
  if constexpr (Dim == 2) {
    const auto& r = spherical_coords[radial_coord];
    const auto& phi = spherical_coords[polar_coord];
    get<radial_coord, 0>(inv_jacobian_matrix) = cos(phi);
    get<radial_coord, 1>(inv_jacobian_matrix) = sin(phi);
    get<polar_coord, 0>(inv_jacobian_matrix) = -r * sin(phi);
    get<polar_coord, 1>(inv_jacobian_matrix) = r * cos(phi);
  } else {
    const auto& r = spherical_coords[radial_coord];
    const auto [x, y, z] = operator()(spherical_coords);
    get<radial_coord, 0>(inv_jacobian_matrix) = x / r;
    get<radial_coord, 1>(inv_jacobian_matrix) = y / r;
    get<radial_coord, 2>(inv_jacobian_matrix) = z / r;
    const auto rho_square = square(x) + square(y);
    const auto denominator = square(r) * sqrt(rho_square);
    get<polar_coord, 0>(inv_jacobian_matrix) = x * z / denominator;
    get<polar_coord, 1>(inv_jacobian_matrix) = y * z / denominator;
    get<polar_coord, 2>(inv_jacobian_matrix) = -rho_square / denominator;
    get<azimuth_coord, 0>(inv_jacobian_matrix) = -y / rho_square;
    get<azimuth_coord, 1>(inv_jacobian_matrix) = x / rho_square;
    get<azimuth_coord, 2>(inv_jacobian_matrix) = 0.;
  }
  return inv_jacobian_matrix;
}

template <size_t Dim>
void SphericalToCartesian<Dim>::pup(PUP::er& p) {
  size_t version = 0;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
}

template <size_t Dim>
bool operator!=(const SphericalToCartesian<Dim>& lhs,
                const SphericalToCartesian<Dim>& rhs) {
  return not(lhs == rhs);
}

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_DIM(_, data)                                        \
  template class SphericalToCartesian<DIM(data)>;                       \
  template bool operator==(const SphericalToCartesian<DIM(data)>& lhs,  \
                           const SphericalToCartesian<DIM(data)>& rhs); \
  template bool operator!=(const SphericalToCartesian<DIM(data)>& lhs,  \
                           const SphericalToCartesian<DIM(data)>& rhs);

#define INSTANTIATE_DTYPE(_, data)                                     \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data)> \
  SphericalToCartesian<DIM(data)>::operator()(                         \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const;  \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),   \
                    Frame::NoFrame>                                    \
  SphericalToCartesian<DIM(data)>::jacobian(                           \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const;  \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),   \
                    Frame::NoFrame>                                    \
  SphericalToCartesian<DIM(data)>::inv_jacobian(                       \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_DIM, (2, 3))
GENERATE_INSTANTIATIONS(INSTANTIATE_DTYPE, (2, 3),
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))

#undef DIM
#undef DTYPE
#undef INSTANTIATE_DIM
#undef INSTANTIATE_DTYPE
}  // namespace domain::CoordinateMaps
