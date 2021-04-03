// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Wedge.hpp"

#include <cmath>
#include <cstddef>
#include <pup.h>

#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain::CoordinateMaps {

template <size_t Dim>
Wedge<Dim>::Wedge(const double radius_inner, const double radius_outer,
                  const double sphericity_inner, const double sphericity_outer,
                  OrientationMap<Dim> orientation_of_wedge,
                  const bool with_equiangular_map,
                  const WedgeHalves halves_to_use,
                  const bool with_logarithmic_map) noexcept
    : radius_inner_(radius_inner),
      radius_outer_(radius_outer),
      sphericity_inner_(sphericity_inner),
      sphericity_outer_(sphericity_outer),
      orientation_of_wedge_(std::move(orientation_of_wedge)),
      with_equiangular_map_(with_equiangular_map),
      halves_to_use_(halves_to_use),
      with_logarithmic_map_(with_logarithmic_map) {
  const double sqrt_dim = sqrt(double{Dim});
  ASSERT(radius_inner > 0.0,
         "The radius of the inner surface must be greater than zero.");
  ASSERT(sphericity_inner >= 0.0 and sphericity_inner <= 1.0,
         "Sphericity of the inner surface must be between 0 and 1");
  ASSERT(sphericity_outer >= 0.0 and sphericity_outer <= 1.0,
         "Sphericity of the outer surface must be between 0 and 1");
  ASSERT(radius_outer > radius_inner,
         "The radius of the outer surface must be greater than the radius of "
         "the inner surface.");
  ASSERT(radius_outer *
                 ((1.0 - sphericity_outer_) / sqrt_dim + sphericity_outer_) >
             radius_inner *
                 ((1.0 - sphericity_inner) / sqrt_dim + sphericity_inner),
         "The arguments passed into the constructor for Wedge result in an "
         "object where the "
         "outer surface is pierced by the inner surface.");
  ASSERT(not with_logarithmic_map_ or
             (with_logarithmic_map_ and sphericity_inner_ == 1.0 and
              sphericity_outer_ == 1.0),
         "The logarithmic map is only supported for spherical wedges.");
  ASSERT(
      get(determinant(discrete_rotation_jacobian(orientation_of_wedge_))) > 0.0,
      "Wedge rotations must be done in such a manner that the sign of "
      "the determinant of the discrete rotation is positive. This is to "
      "preserve handedness of the coordinates.");
  if (with_logarithmic_map_) {
    scaled_frustum_zero_ = 0.0;
    sphere_zero_ = 0.5 * (log(radius_outer * radius_inner));
    scaled_frustum_rate_ = 0.0;
    sphere_rate_ = 0.5 * (log(radius_outer / radius_inner));
  } else {
    scaled_frustum_zero_ = 0.5 / sqrt_dim *
                           ((1.0 - sphericity_outer_) * radius_outer +
                            (1.0 - sphericity_inner) * radius_inner);
    sphere_zero_ = 0.5 * (sphericity_outer_ * radius_outer +
                          sphericity_inner * radius_inner);
    scaled_frustum_rate_ = 0.5 / sqrt_dim *
                           ((1.0 - sphericity_outer_) * radius_outer -
                            (1.0 - sphericity_inner) * radius_inner);
    sphere_rate_ = 0.5 * (sphericity_outer_ * radius_outer -
                          sphericity_inner * radius_inner);
  }
}

template <size_t Dim>
template <typename T>
tt::remove_cvref_wrap_t<T> Wedge<Dim>::default_physical_z(
    const T& zeta, const T& one_over_rho) const noexcept {
  if (with_logarithmic_map_) {
    return exp(sphere_zero_ + sphere_rate_ * zeta) * one_over_rho;
  }

  // Using auto keeps this as a blaze expression.
  const auto zeta_coefficient =
      (scaled_frustum_rate_ + sphere_rate_ * one_over_rho);
  const auto z_zero = (scaled_frustum_zero_ + sphere_zero_ * one_over_rho);
  return z_zero + zeta_coefficient * zeta;
}

template <size_t Dim>
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, Dim> Wedge<Dim>::operator()(
    const std::array<T, Dim>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  // Radial coordinate
  const ReturnType& zeta = source_coords[radial_coord];

  // Polar angle
  ReturnType xi = source_coords[polar_coord];
  if (halves_to_use_ == WedgeHalves::UpperOnly) {
    xi += 1.0;
    xi *= 0.5;
  } else if (halves_to_use_ == WedgeHalves::LowerOnly) {
    xi -= 1.0;
    xi *= 0.5;
  }

  std::array<ReturnType, Dim - 1> cap{};
  cap[0] = with_equiangular_map_ ? tan(M_PI_4 * xi) : xi;
  ReturnType one_over_rho = 1.0 + square(cap[0]);
  if constexpr (Dim == 3) {
    // Azimuthal angle
    const ReturnType& eta = source_coords[azimuth_coord];
    cap[1] = with_equiangular_map_ ? tan(M_PI_4 * eta) : eta;
    one_over_rho += square(cap[1]);
  }
  one_over_rho = 1. / sqrt(one_over_rho);

  std::array<ReturnType, Dim> physical_coords{};
  physical_coords[radial_coord] = default_physical_z(zeta, one_over_rho);
  physical_coords[polar_coord] = physical_coords[radial_coord] * cap[0];
  if constexpr (Dim == 3) {
    physical_coords[azimuth_coord] = physical_coords[radial_coord] * cap[1];
  }

  return discrete_rotation(orientation_of_wedge_, std::move(physical_coords));
}

template <size_t Dim>
std::optional<std::array<double, Dim>> Wedge<Dim>::inverse(
    const std::array<double, Dim>& target_coords) const noexcept {
  const std::array<double, Dim> physical_coords =
      discrete_rotation(orientation_of_wedge_.inverse_map(), target_coords);

  if (physical_coords[radial_coord] < 0.0 or
      equal_within_roundoff(physical_coords[radial_coord], 0.0)) {
    return std::nullopt;
  }

  std::array<double, Dim - 1> cap{};
  cap[0] = physical_coords[polar_coord] / physical_coords[radial_coord];
  double one_over_rho = 1.0 + square(cap[0]);
  if constexpr (Dim == 3) {
    cap[1] = physical_coords[azimuth_coord] / physical_coords[radial_coord];
    one_over_rho += square(cap[1]);
  }
  one_over_rho = 1. / sqrt(one_over_rho);
  const double zeta_coefficient =
      (scaled_frustum_rate_ + sphere_rate_ * one_over_rho);
  // If -sphere_rate_/scaled_frustum_rate_ > 1, then
  // there exists a cone in x,y,z space given by the surface
  // zeta_coefficient=0; the map is singular on this surface.
  // We return nullopt if we are on or outside this cone.
  // If scaled_frustum_rate_ > 0, then outside the cone
  // corresponds to zeta_coefficient > 0, and if scaled_frustum_rate_
  // < 0, then outside the cone corresponds to zeta_coefficient < 0.
  // We test in two cases, and avoid division.
  if ((scaled_frustum_rate_ > 0.0 and scaled_frustum_rate_ < -sphere_rate_ and
       zeta_coefficient > 0.0) or
      (scaled_frustum_rate_ < 0.0 and scaled_frustum_rate_ > -sphere_rate_ and
       zeta_coefficient < 0.0) or
      equal_within_roundoff(zeta_coefficient, 0.0)) {
    return std::nullopt;
  }
  const auto z_zero = (scaled_frustum_zero_ + sphere_zero_ * one_over_rho);
  // Radial coordinate
  const double zeta =
      with_logarithmic_map_
          ? (log(physical_coords[radial_coord] / one_over_rho) - sphere_zero_) /
                sphere_rate_
          : (physical_coords[radial_coord] - z_zero) / zeta_coefficient;
  // Polar angle
  double xi = with_equiangular_map_ ? atan(cap[0]) / M_PI_4 : cap[0];
  if (halves_to_use_ == WedgeHalves::UpperOnly) {
    xi *= 2.0;
    xi -= 1.0;
  } else if (halves_to_use_ == WedgeHalves::LowerOnly) {
    xi *= 2.0;
    xi += 1.0;
  }
  std::array<double, Dim> logical_coords{};
  logical_coords[radial_coord] = zeta;
  logical_coords[polar_coord] = xi;
  if constexpr (Dim == 3) {
    logical_coords[azimuth_coord] =
        with_equiangular_map_ ? atan(cap[1]) / M_PI_4 : cap[1];
  }
  return logical_coords;
}

template <size_t Dim>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame> Wedge<Dim>::jacobian(
    const std::array<T, Dim>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  // Radial coordinate
  const ReturnType& zeta = source_coords[radial_coord];

  // Polar angle
  ReturnType xi = source_coords[polar_coord];
  if (halves_to_use_ == WedgeHalves::UpperOnly) {
    xi += 1.0;
    xi *= 0.5;
  } else if (halves_to_use_ == WedgeHalves::LowerOnly) {
    xi -= 1.0;
    xi *= 0.5;
  }
  std::array<ReturnType, Dim - 1> cap{};
  std::array<ReturnType, Dim - 1> cap_deriv{};
  cap[0] = with_equiangular_map_ ? tan(M_PI_4 * xi) : xi;
  cap_deriv[0] = with_equiangular_map_ ? M_PI_4 * (1.0 + square(cap[0]))
                                       : make_with_value<ReturnType>(xi, 1.0);
  ReturnType one_over_rho = 1.0 + square(cap[0]);
  if constexpr (Dim == 3) {
    // Azimuthal angle
    const ReturnType& eta = source_coords[azimuth_coord];
    cap[1] = with_equiangular_map_ ? tan(M_PI_4 * eta) : eta;
    cap_deriv[1] = with_equiangular_map_ ? M_PI_4 * (1.0 + square(cap[1]))
                                         : make_with_value<ReturnType>(xi, 1.0);
    one_over_rho += square(cap[1]);
  }
  one_over_rho = 1. / sqrt(one_over_rho);

  const ReturnType one_over_rho_cubed = pow<3>(one_over_rho);
  const ReturnType physical_z = default_physical_z(zeta, one_over_rho);
  const ReturnType s_factor =
      with_logarithmic_map_
          ? ReturnType{exp(sphere_zero_ + sphere_rate_ * zeta)}
          : ReturnType{sphere_zero_ + sphere_rate_ * zeta};

  auto jacobian_matrix =
      make_with_value<tnsr::Ij<ReturnType, Dim, Frame::NoFrame>>(xi, 0.0);

  // Derivative by polar angle
  std::array<ReturnType, Dim> dxyz_dxi{};
  dxyz_dxi[radial_coord] =
      -s_factor * cap[0] * cap_deriv[0] * one_over_rho_cubed;
  dxyz_dxi[polar_coord] =
      cap[0] * dxyz_dxi[radial_coord] + cap_deriv[0] * physical_z;
  if constexpr (Dim == 3) {
    dxyz_dxi[azimuth_coord] = cap[1] * dxyz_dxi[radial_coord];
  }
  // Implement Scalings:
  if (halves_to_use_ != WedgeHalves::Both) {
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(dxyz_dxi, d) *= 0.5;
    }
  }
  std::array<ReturnType, Dim> dX_dlogical =
      discrete_rotation(orientation_of_wedge_, std::move(dxyz_dxi));
  get<0, polar_coord>(jacobian_matrix) = dX_dlogical[0];
  get<1, polar_coord>(jacobian_matrix) = dX_dlogical[1];
  if constexpr (Dim == 3) {
    get<2, polar_coord>(jacobian_matrix) = dX_dlogical[2];
  }

  // Derivative by azimuthal angle
  if constexpr (Dim == 3) {
    std::array<ReturnType, Dim> dxyz_deta{};
    dxyz_deta[radial_coord] =
        -s_factor * cap[1] * cap_deriv[1] * one_over_rho_cubed;
    dxyz_deta[azimuth_coord] =
        cap[1] * dxyz_deta[radial_coord] + cap_deriv[1] * physical_z;
    dxyz_deta[polar_coord] = cap[0] * dxyz_deta[radial_coord];
    dX_dlogical =
        discrete_rotation(orientation_of_wedge_, std::move(dxyz_deta));
    get<0, azimuth_coord>(jacobian_matrix) = dX_dlogical[0];
    get<1, azimuth_coord>(jacobian_matrix) = dX_dlogical[1];
    get<2, azimuth_coord>(jacobian_matrix) = dX_dlogical[2];
  }

  // Derivative by radial coordinate
  std::array<ReturnType, Dim> dxyz_dzeta{};
  if (with_logarithmic_map_) {
    dxyz_dzeta[radial_coord] = s_factor * sphere_rate_ * one_over_rho;
  } else {
    dxyz_dzeta[radial_coord] =
        scaled_frustum_rate_ + sphere_rate_ * one_over_rho;
  }
  dxyz_dzeta[polar_coord] = cap[0] * dxyz_dzeta[radial_coord];
  if constexpr (Dim == 3) {
    dxyz_dzeta[azimuth_coord] = cap[1] * dxyz_dzeta[radial_coord];
  }
  dX_dlogical = discrete_rotation(orientation_of_wedge_, std::move(dxyz_dzeta));
  get<0, radial_coord>(jacobian_matrix) = dX_dlogical[0];
  get<1, radial_coord>(jacobian_matrix) = dX_dlogical[1];
  if constexpr (Dim == 3) {
    get<2, radial_coord>(jacobian_matrix) = dX_dlogical[2];
  }
  return jacobian_matrix;
}

template <size_t Dim>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame>
Wedge<Dim>::inv_jacobian(
    const std::array<T, Dim>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  // Radial coordinate
  const ReturnType& zeta = source_coords[radial_coord];

  // Polar angle
  ReturnType xi = source_coords[polar_coord];
  if (halves_to_use_ == WedgeHalves::UpperOnly) {
    xi += 1.0;
    xi *= 0.5;
  } else if (halves_to_use_ == WedgeHalves::LowerOnly) {
    xi -= 1.0;
    xi *= 0.5;
  }
  std::array<ReturnType, Dim> cap{};
  std::array<ReturnType, Dim> cap_deriv{};
  cap[0] = with_equiangular_map_ ? tan(M_PI_4 * xi) : xi;
  cap_deriv[0] = with_equiangular_map_ ? M_PI_4 * (1.0 + square(cap[0]))
                                       : make_with_value<ReturnType>(xi, 1.0);
  ReturnType one_over_rho = 1.0 + square(cap[0]);
  if constexpr (Dim == 3) {
    // Azimuthal angle
    const ReturnType& eta = source_coords[azimuth_coord];
    cap[1] = with_equiangular_map_ ? tan(M_PI_4 * eta) : eta;
    cap_deriv[1] = with_equiangular_map_ ? M_PI_4 * (1.0 + square(cap[1]))
                                         : make_with_value<ReturnType>(xi, 1.0);
    one_over_rho += square(cap[1]);
  }
  one_over_rho = 1. / sqrt(one_over_rho);

  const ReturnType one_over_rho_cubed = pow<3>(one_over_rho);
  const ReturnType one_over_physical_z =
      1.0 / default_physical_z(zeta, one_over_rho);
  const ReturnType scaled_z_frustum =
      scaled_frustum_zero_ + scaled_frustum_rate_ * zeta;
  const ReturnType s_factor_over_rho_cubed =
      with_logarithmic_map_
          ? ReturnType{exp(sphere_zero_ + sphere_rate_ * zeta) *
                       one_over_rho_cubed}
          : ReturnType{(sphere_zero_ + sphere_rate_ * zeta) *
                       one_over_rho_cubed};
  const ReturnType one_over_dz_dzeta =
      with_logarithmic_map_
          ? ReturnType{1.0 / (exp(sphere_zero_ + sphere_rate_ * zeta) *
                              sphere_rate_ * one_over_rho)}
          : ReturnType{1.0 /
                       (scaled_frustum_rate_ + sphere_rate_ * one_over_rho)};
  const ReturnType dzeta_factor = one_over_physical_z * one_over_dz_dzeta;

  auto inv_jacobian_matrix =
      make_with_value<tnsr::Ij<ReturnType, Dim, Frame::NoFrame>>(xi, 0.0);

  // Derivatives of polar angle
  std::array<ReturnType, Dim> dxi_dxyz{};
  dxi_dxyz[polar_coord] = one_over_physical_z / cap_deriv[0];
  // Implement Scalings:
  if (halves_to_use_ != WedgeHalves::Both) {
    dxi_dxyz[polar_coord] *= 2.0;
  }
  dxi_dxyz[radial_coord] = -cap[0] * dxi_dxyz[polar_coord];
  if constexpr (Dim == 3) {
    dxi_dxyz[azimuth_coord] = make_with_value<ReturnType>(xi, 0.0);
  }
  std::array<ReturnType, Dim> dlogical_dX =
      discrete_rotation(orientation_of_wedge_, std::move(dxi_dxyz));
  get<polar_coord, 0>(inv_jacobian_matrix) = dlogical_dX[0];
  get<polar_coord, 1>(inv_jacobian_matrix) = dlogical_dX[1];
  if constexpr (Dim == 3) {
    get<polar_coord, 2>(inv_jacobian_matrix) = dlogical_dX[2];
  }

  // Derivatives of azimuthal angle
  if constexpr (Dim == 3) {
    std::array<ReturnType, Dim> deta_dxyz{};
    deta_dxyz[polar_coord] = make_with_value<ReturnType>(xi, 0.0);
    deta_dxyz[azimuth_coord] = one_over_physical_z / cap_deriv[1];
    deta_dxyz[radial_coord] = -cap[1] * deta_dxyz[azimuth_coord];
    dlogical_dX =
        discrete_rotation(orientation_of_wedge_, std::move(deta_dxyz));
    get<azimuth_coord, 0>(inv_jacobian_matrix) = dlogical_dX[0];
    get<azimuth_coord, 1>(inv_jacobian_matrix) = dlogical_dX[1];
    get<azimuth_coord, 2>(inv_jacobian_matrix) = dlogical_dX[2];
  }

  // Derivatives of radial coordinate
  std::array<ReturnType, Dim> dzeta_dxyz{};
  dzeta_dxyz[radial_coord] =
      dzeta_factor * (scaled_z_frustum + s_factor_over_rho_cubed);
  dzeta_dxyz[polar_coord] = dzeta_factor * cap[0] * s_factor_over_rho_cubed;
  if constexpr (Dim == 3) {
    dzeta_dxyz[azimuth_coord] = dzeta_factor * cap[1] * s_factor_over_rho_cubed;
  }
  dlogical_dX = discrete_rotation(orientation_of_wedge_, std::move(dzeta_dxyz));
  get<radial_coord, 0>(inv_jacobian_matrix) = dlogical_dX[0];
  get<radial_coord, 1>(inv_jacobian_matrix) = dlogical_dX[1];
  if constexpr (Dim == 3) {
    get<radial_coord, 2>(inv_jacobian_matrix) = dlogical_dX[2];
  }
  return inv_jacobian_matrix;
}

template <size_t Dim>
void Wedge<Dim>::pup(PUP::er& p) noexcept {
  p | radius_inner_;
  p | radius_outer_;
  p | sphericity_inner_;
  p | sphericity_outer_;
  p | orientation_of_wedge_;
  p | with_equiangular_map_;
  p | halves_to_use_;
  p | with_logarithmic_map_;
  p | scaled_frustum_zero_;
  p | sphere_zero_;
  p | scaled_frustum_rate_;
  p | sphere_rate_;
}

template <size_t Dim>
bool operator==(const Wedge<Dim>& lhs, const Wedge<Dim>& rhs) noexcept {
  return lhs.radius_inner_ == rhs.radius_inner_ and
         lhs.radius_outer_ == rhs.radius_outer_ and
         lhs.orientation_of_wedge_ == rhs.orientation_of_wedge_ and
         lhs.with_equiangular_map_ == rhs.with_equiangular_map_ and
         lhs.halves_to_use_ == rhs.halves_to_use_ and
         lhs.with_logarithmic_map_ == rhs.with_logarithmic_map_ and
         lhs.sphericity_inner_ == rhs.sphericity_inner_ and
         lhs.sphericity_outer_ == rhs.sphericity_outer_ and
         lhs.scaled_frustum_zero_ == rhs.scaled_frustum_zero_ and
         lhs.sphere_zero_ == rhs.sphere_zero_ and
         lhs.scaled_frustum_rate_ == rhs.scaled_frustum_rate_ and
         lhs.sphere_rate_ == rhs.sphere_rate_;
}

template <size_t Dim>
bool operator!=(const Wedge<Dim>& lhs, const Wedge<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_DIM(_, data)                                  \
  template class Wedge<DIM(data)>;                                \
  template bool operator==(const Wedge<DIM(data)>& lhs,           \
                           const Wedge<DIM(data)>& rhs) noexcept; \
  template bool operator!=(const Wedge<DIM(data)>& lhs,           \
                           const Wedge<DIM(data)>& rhs) noexcept;

#define INSTANTIATE_DTYPE(_, data)                                             \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data)>         \
  Wedge<DIM(data)>::operator()(                                                \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const noexcept; \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),           \
                    Frame::NoFrame>                                            \
  Wedge<DIM(data)>::jacobian(                                                  \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const noexcept; \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),           \
                    Frame::NoFrame>                                            \
  Wedge<DIM(data)>::inv_jacobian(                                              \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const noexcept;

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
