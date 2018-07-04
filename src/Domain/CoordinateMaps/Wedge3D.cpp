// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Wedge3D.hpp"

#include <boost/none.hpp>
#include <cmath>
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/SegmentId.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain {
namespace CoordinateMaps {

Wedge3D::Wedge3D(const double radius_inner, const double radius_outer,
                 const OrientationMap<3> orientation_of_wedge,
                 const double sphericity_inner, const double sphericity_outer,
                 const bool with_equiangular_map,
                 const WedgeHalves halves_to_use,
                 const bool with_logarithmic_map) noexcept
    : radius_inner_(radius_inner),
      radius_outer_(radius_outer),
      orientation_of_wedge_(orientation_of_wedge),
      sphericity_inner_(sphericity_inner),
      sphericity_outer_(sphericity_outer),
      with_equiangular_map_(with_equiangular_map),
      halves_to_use_(halves_to_use),
      with_logarithmic_map_(with_logarithmic_map) {
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
                 ((1.0 - sphericity_outer_) / sqrt(3.0) + sphericity_outer_) >
             radius_inner *
                 ((1.0 - sphericity_inner) / sqrt(3.0) + sphericity_inner),
         "The arguments passed into the constructor for Wedge3D result in an "
         "object where the "
         "outer surface is pierced by the inner surface.");
  ASSERT(not with_logarithmic_map_ or
             (with_logarithmic_map_ and sphericity_inner_ == 1.0 and
              sphericity_outer_ == 1.0),
         "The logarithmic map is only supported for spherical wedges.");
  if (with_logarithmic_map_) {
    scaled_frustum_zero_ = 0.0;
    sphere_zero_ = 0.5 * (log(radius_outer * radius_inner));
    scaled_frustum_rate_ = 0.0;
    sphere_rate_ = 0.5 * (log(radius_outer / radius_inner));
  } else {
    scaled_frustum_zero_ = 0.5 / sqrt(3.0) *
                           ((1.0 - sphericity_outer_) * radius_outer +
                            (1.0 - sphericity_inner) * radius_inner);
    sphere_zero_ = 0.5 * (sphericity_outer_ * radius_outer +
                          sphericity_inner * radius_inner);
    scaled_frustum_rate_ = 0.5 / sqrt(3.0) *
                           ((1.0 - sphericity_outer_) * radius_outer -
                            (1.0 - sphericity_inner) * radius_inner);
    sphere_rate_ = 0.5 * (sphericity_outer_ * radius_outer -
                          sphericity_inner * radius_inner);
  }
}

template <typename T>
tt::remove_cvref_wrap_t<T> Wedge3D::default_physical_z(
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

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> Wedge3D::operator()(
    const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  ReturnType xi = source_coords[0];
  if (halves_to_use_ == WedgeHalves::UpperOnly) {
    xi += 1.0;
    xi *= 0.5;
  } else if (halves_to_use_ == WedgeHalves::LowerOnly) {
    xi -= 1.0;
    xi *= 0.5;
  }
  const ReturnType& eta = source_coords[1];

  const ReturnType cap_xi = with_equiangular_map_ ? tan(M_PI_4 * xi) : xi;
  const ReturnType cap_eta = with_equiangular_map_ ? tan(M_PI_4 * eta) : eta;
  const ReturnType& zeta = source_coords[2];
  const ReturnType one_over_rho =
      1.0 / sqrt(1.0 + square(cap_xi) + square(cap_eta));

  ReturnType physical_z = default_physical_z(zeta, one_over_rho);
  ReturnType physical_x = physical_z * cap_xi;
  ReturnType physical_y = physical_z * cap_eta;

  std::array<ReturnType, 3> physical_coords{
      {std::move(physical_x), std::move(physical_y), std::move(physical_z)}};
  return discrete_rotation(orientation_of_wedge_, std::move(physical_coords));
}

boost::optional<std::array<double, 3>> Wedge3D::inverse(
    const std::array<double, 3>& target_coords) const noexcept {
  const std::array<double, 3> physical_coords =
      discrete_rotation(orientation_of_wedge_.inverse_map(), target_coords);
  const double& physical_x = physical_coords[0];
  const double& physical_y = physical_coords[1];
  const double& physical_z = physical_coords[2];

  if (physical_z < 0.0 or equal_within_roundoff(physical_z, 0.0)) {
    return boost::none;
  }

  const double cap_xi = physical_x / physical_z;
  const double cap_eta = physical_y / physical_z;
  const double one_over_rho =
      1.0 / sqrt(1.0 + square(cap_xi) + square(cap_eta));
  const double zeta_coefficient =
      (scaled_frustum_rate_ + sphere_rate_ * one_over_rho);
  // If -sphere_rate_/scaled_frustum_rate_ > 1, then
  // there exists a cone in x,y,z space given by the surface
  // zeta_coefficient=0; the map is singular on this surface.
  // We return boost::none if we are on or outside this cone.
  // If scaled_frustum_rate_ > 0, then outside the cone
  // corresponds to zeta_coefficient > 0, and if scaled_frustum_rate_
  // < 0, then outside the cone corresponds to zeta_coefficient < 0.
  // We test in two cases, and avoid division.
  if ((scaled_frustum_rate_ > 0.0 and scaled_frustum_rate_ < -sphere_rate_ and
       zeta_coefficient > 0.0) or
      (scaled_frustum_rate_ < 0.0 and scaled_frustum_rate_ > -sphere_rate_ and
       zeta_coefficient < 0.0) or
      equal_within_roundoff(zeta_coefficient, 0.0)) {
    return boost::none;
  }
  const auto z_zero = (scaled_frustum_zero_ + sphere_zero_ * one_over_rho);
  double zeta;
  if (with_logarithmic_map_) {
    zeta = (log(physical_z * sqrt(1.0 + square(cap_xi) + square(cap_eta))) -
            sphere_zero_) /
           sphere_rate_;
  } else {
    zeta = (physical_z - z_zero) / zeta_coefficient;
  }
  double xi = with_equiangular_map_ ? atan(cap_xi) / M_PI_4 : cap_xi;
  double eta =
      with_equiangular_map_ ? atan(cap_eta) / M_PI_4 : cap_eta;
  if (halves_to_use_ == WedgeHalves::UpperOnly) {
    xi *= 2.0;
    xi -= 1.0;
  } else if (halves_to_use_ == WedgeHalves::LowerOnly) {
    xi *= 2.0;
    xi += 1.0;
  }
  return std::array<double, 3>{
      {xi, eta, zeta}};
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> Wedge3D::jacobian(
    const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  ReturnType xi = source_coords[0];
  if (halves_to_use_ == WedgeHalves::UpperOnly) {
    xi += 1.0;
    xi *= 0.5;
  } else if (halves_to_use_ == WedgeHalves::LowerOnly) {
    xi -= 1.0;
    xi *= 0.5;
  }
  const ReturnType& eta = source_coords[1];
  const ReturnType& zeta = source_coords[2];
  const ReturnType& cap_xi = with_equiangular_map_ ? tan(M_PI_4 * xi) : xi;
  const ReturnType& cap_eta = with_equiangular_map_ ? tan(M_PI_4 * eta) : eta;
  const ReturnType cap_xi_deriv = with_equiangular_map_
                                      ? M_PI_4 * (1.0 + square(cap_xi))
                                      : make_with_value<ReturnType>(xi, 1.0);
  const ReturnType cap_eta_deriv = with_equiangular_map_
                                       ? M_PI_4 * (1.0 + square(cap_eta))
                                       : make_with_value<ReturnType>(eta, 1.0);

  const ReturnType one_over_rho =
      1.0 / sqrt(1.0 + square(cap_xi) + square(cap_eta));
  const ReturnType one_over_rho_cubed = pow<3>(one_over_rho);
  const ReturnType physical_z = default_physical_z(zeta, one_over_rho);
  const ReturnType s_factor =
      with_logarithmic_map_
          ? ReturnType{exp(sphere_zero_ + sphere_rate_ * zeta)}
          : ReturnType{sphere_zero_ + sphere_rate_ * zeta};

  auto jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);

  ReturnType dz_dxi = -s_factor * cap_xi * cap_xi_deriv * one_over_rho_cubed;
  ReturnType dx_dxi = cap_xi * dz_dxi + cap_xi_deriv * physical_z;
  ReturnType dy_dxi = cap_eta * dz_dxi;
  // Implement Scalings:
  if (halves_to_use_ != WedgeHalves::Both) {
    dz_dxi *= 0.5;
    dy_dxi *= 0.5;
    dx_dxi *= 0.5;
  }
  std::array<ReturnType, 3> dX_dlogical = discrete_rotation(
      orientation_of_wedge_,
      std::array<ReturnType, 3>{
          {std::move(dx_dxi), std::move(dy_dxi), std::move(dz_dxi)}});
  get<0, 0>(jacobian_matrix) = dX_dlogical[0];
  get<1, 0>(jacobian_matrix) = dX_dlogical[1];
  get<2, 0>(jacobian_matrix) = dX_dlogical[2];

  // dz_deta
  dX_dlogical[2] = -s_factor * cap_eta * cap_eta_deriv * one_over_rho_cubed;
  // dy_deta
  dX_dlogical[1] = cap_eta * dX_dlogical[2] + cap_eta_deriv * physical_z;
  // dx_deta
  dX_dlogical[0] = cap_xi * dX_dlogical[2];
  dX_dlogical =
      discrete_rotation(orientation_of_wedge_, std::move(dX_dlogical));
  get<0, 1>(jacobian_matrix) = dX_dlogical[0];
  get<1, 1>(jacobian_matrix) = dX_dlogical[1];
  get<2, 1>(jacobian_matrix) = dX_dlogical[2];

  // dz_dzeta
  if (with_logarithmic_map_) {
    dX_dlogical[2] = s_factor * sphere_rate_ * one_over_rho;
  } else {
    dX_dlogical[2] = scaled_frustum_rate_ + sphere_rate_ * one_over_rho;
  }
  // dx_dzeta
  dX_dlogical[0] = cap_xi * dX_dlogical[2];
  // dy_dzeta
  dX_dlogical[1] = cap_eta * dX_dlogical[2];
  dX_dlogical =
      discrete_rotation(orientation_of_wedge_, std::move(dX_dlogical));
  get<0, 2>(jacobian_matrix) = dX_dlogical[0];
  get<1, 2>(jacobian_matrix) = dX_dlogical[1];
  get<2, 2>(jacobian_matrix) = dX_dlogical[2];
  return jacobian_matrix;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> Wedge3D::inv_jacobian(
    const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  ReturnType xi = source_coords[0];
  if (halves_to_use_ == WedgeHalves::UpperOnly) {
    xi += make_with_value<ReturnType>(xi, 1.0);
    xi *= 0.5;
  } else if (halves_to_use_ == WedgeHalves::LowerOnly) {
    xi += make_with_value<ReturnType>(xi, -1.0);
    xi *= 0.5;
  }

  const ReturnType& eta = source_coords[1];
  const ReturnType& zeta = source_coords[2];
  const ReturnType cap_xi = with_equiangular_map_ ? tan(M_PI_4 * xi) : xi;
  const ReturnType cap_eta = with_equiangular_map_ ? tan(M_PI_4 * eta) : eta;
  const ReturnType cap_xi_deriv = with_equiangular_map_
                                      ? M_PI_4 * (1.0 + square(cap_xi))
                                      : make_with_value<ReturnType>(xi, 1.0);
  const ReturnType cap_eta_deriv = with_equiangular_map_
                                       ? M_PI_4 * (1.0 + square(cap_eta))
                                       : make_with_value<ReturnType>(eta, 1.0);

  const ReturnType one_over_rho =
      1.0 / sqrt(1.0 + square(cap_xi) + square(cap_eta));
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
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);

  ReturnType dxi_dx = one_over_physical_z / cap_xi_deriv;
  // Implement Scalings:
  if (halves_to_use_ != WedgeHalves::Both) {
    dxi_dx *= 2.0;
  }
  auto dxi_dy = make_with_value<ReturnType>(xi, 0.0);
  ReturnType dxi_dz = -cap_xi * dxi_dx;

  std::array<ReturnType, 3> dlogical_dX = discrete_rotation(
      orientation_of_wedge_,
      std::array<ReturnType, 3>{
          {std::move(dxi_dx), std::move(dxi_dy), std::move(dxi_dz)}});
  get<0, 0>(inv_jacobian_matrix) = dlogical_dX[0];
  get<0, 1>(inv_jacobian_matrix) = dlogical_dX[1];
  get<0, 2>(inv_jacobian_matrix) = dlogical_dX[2];

  // deta_dx
  dlogical_dX[0] = 0.0;
  // deta_dy
  dlogical_dX[1] = one_over_physical_z / cap_eta_deriv;
  // deta_dz
  dlogical_dX[2] = -cap_eta * dlogical_dX[1];
  dlogical_dX =
      discrete_rotation(orientation_of_wedge_, std::move(dlogical_dX));
  get<1, 0>(inv_jacobian_matrix) = dlogical_dX[0];
  get<1, 1>(inv_jacobian_matrix) = dlogical_dX[1];
  get<1, 2>(inv_jacobian_matrix) = dlogical_dX[2];

  // dzeta_dx
  dlogical_dX[0] = dzeta_factor * cap_xi * s_factor_over_rho_cubed;
  // dzeta_dy
  dlogical_dX[1] = dzeta_factor * cap_eta * s_factor_over_rho_cubed;
  // dzeta_dz
  dlogical_dX[2] = dzeta_factor * (scaled_z_frustum + s_factor_over_rho_cubed);
  dlogical_dX =
      discrete_rotation(orientation_of_wedge_, std::move(dlogical_dX));
  get<2, 0>(inv_jacobian_matrix) = dlogical_dX[0];
  get<2, 1>(inv_jacobian_matrix) = dlogical_dX[1];
  get<2, 2>(inv_jacobian_matrix) = dlogical_dX[2];
  return inv_jacobian_matrix;
}

void Wedge3D::pup(PUP::er& p) noexcept {
  p | radius_inner_;
  p | radius_outer_;
  p | orientation_of_wedge_;
  p | sphericity_inner_;
  p | sphericity_outer_;
  p | with_equiangular_map_;
  p | halves_to_use_;
  p | with_logarithmic_map_;
  p | scaled_frustum_zero_;
  p | sphere_zero_;
  p | scaled_frustum_rate_;
  p | sphere_rate_;
}

bool operator==(const Wedge3D& lhs, const Wedge3D& rhs) noexcept {
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

bool operator!=(const Wedge3D& lhs, const Wedge3D& rhs) noexcept {
  return not(lhs == rhs);
}

// Explicit instantiations
/// \cond
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3> Wedge3D::      \
  operator()(const std::array<DTYPE(data), 3>& source_coords) const noexcept; \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>  \
  Wedge3D::jacobian(const std::array<DTYPE(data), 3>& source_coords)          \
      const noexcept;                                                         \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>  \
  Wedge3D::inv_jacobian(const std::array<DTYPE(data), 3>& source_coords)      \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))

#undef DTYPE
#undef INSTANTIATE
/// \endcond
}  // namespace CoordinateMaps
}  // namespace domain
