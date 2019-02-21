// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Wedge2D.hpp"

#include <boost/none.hpp>
#include <cmath>
#include <pup.h>

#include "Domain/OrientationMap.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain {
namespace CoordinateMaps {

Wedge2D::Wedge2D(double radius_inner, double radius_outer,
                 double circularity_inner, double circularity_outer,
                 OrientationMap<2> orientation_of_wedge,
                 bool with_equiangular_map) noexcept
    : radius_inner_(radius_inner),
      radius_outer_(radius_outer),
      circularity_inner_(circularity_inner),
      circularity_outer_(circularity_outer),
      orientation_of_wedge_(orientation_of_wedge),
      with_equiangular_map_(with_equiangular_map) {
  ASSERT(radius_inner > 0.0,
         "The radius of the inner surface must be greater than zero.");
  ASSERT(circularity_inner >= 0.0 and circularity_inner <= 1.0,
         "Circularity of the inner surface must be between 0 and 1");
  ASSERT(circularity_outer >= 0.0 and circularity_outer <= 1.0,
         "Circularity of the outer surface must be between 0 and 1");
  ASSERT(radius_outer > radius_inner,
         "The radius of the outer surface must be greater than the radius of "
         "the inner surface.");
  ASSERT(radius_outer *
                 ((1.0 - circularity_outer) / sqrt(2.0) + circularity_outer) >
             radius_inner *
                 ((1.0 - circularity_inner) / sqrt(2.0) + circularity_inner),
         "The arguments passed into the constructor for Wedge2D result in an "
         "object where the outer surface is pierced by the inner surface.");
  scaled_trapezoid_zero_ =
      0.5 / sqrt(2.0) * ((1.0 - circularity_outer) * radius_outer +
                         (1.0 - circularity_inner) * radius_inner);
  annulus_zero_ = 0.5 * (circularity_outer * radius_outer +
                         circularity_inner * radius_inner);
  scaled_trapezoid_rate_ =
      0.5 / sqrt(2.0) * ((1.0 - circularity_outer) * radius_outer -
                         (1.0 - circularity_inner) * radius_inner);
  annulus_rate_ = 0.5 * (circularity_outer * radius_outer -
                         circularity_inner * radius_inner);
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 2> Wedge2D::operator()(
    const std::array<T, 2>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& xi = source_coords[0];
  const ReturnType& eta = source_coords[1];
  const ReturnType cap_eta = with_equiangular_map_ ? tan(M_PI_4 * eta) : eta;
  ReturnType physical_x =
      scaled_trapezoid_zero_ + scaled_trapezoid_rate_ * xi +
      (annulus_zero_ + annulus_rate_ * xi) / sqrt(1.0 + square(cap_eta));
  ReturnType physical_y = cap_eta * physical_x;
  std::array<ReturnType, 2> physical_coords{
      {std::move(physical_x), std::move(physical_y)}};
  return discrete_rotation(orientation_of_wedge_, std::move(physical_coords));
}

boost::optional<std::array<double, 2>> Wedge2D::inverse(
    const std::array<double, 2>& target_coords) const noexcept {
  const std::array<double, 2> physical_coords =
      discrete_rotation(orientation_of_wedge_.inverse_map(), target_coords);
  const double& physical_x = physical_coords[0];
  const double& physical_y = physical_coords[1];

  if (physical_x < 0.0 or equal_within_roundoff(physical_x, 0.0)) {
    return boost::none;
  }

  const double cap_eta = physical_y / physical_x;
  const double one_over_rho = 1.0 / sqrt(1.0 + square(cap_eta));
  double xi =
      (physical_x - scaled_trapezoid_zero_ - annulus_zero_ * one_over_rho) /
      (scaled_trapezoid_rate_ + annulus_rate_ * one_over_rho);
  double eta =
      with_equiangular_map_ ? atan(cap_eta) / M_PI_4 : cap_eta;
  return std::array<double, 2>{{xi, eta}};
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame> Wedge2D::jacobian(
    const std::array<T, 2>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& xi = source_coords[0];
  const ReturnType& eta = source_coords[1];
  const ReturnType& cap_eta = with_equiangular_map_ ? tan(M_PI_4 * eta) : eta;
  const ReturnType cap_eta_deriv = with_equiangular_map_
                                       ? M_PI_4 * (1.0 + square(cap_eta))
                                       : make_with_value<ReturnType>(eta, 1.0);
  const ReturnType one_over_rho = 1.0 / sqrt(1.0 + square(cap_eta));
  auto jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);

  ReturnType dx_dxi = scaled_trapezoid_rate_ + annulus_rate_ * one_over_rho;
  ReturnType dy_dxi = cap_eta * dx_dxi;
  std::array<ReturnType, 2> dX_dlogical = discrete_rotation(
      orientation_of_wedge_,
      std::array<ReturnType, 2>{{std::move(dx_dxi), std::move(dy_dxi)}});
  get<0, 0>(jacobian_matrix) = dX_dlogical[0];
  get<1, 0>(jacobian_matrix) = dX_dlogical[1];

  // dx_deta
  dX_dlogical[0] = -(annulus_zero_ + annulus_rate_ * xi) * cap_eta *
                   cap_eta_deriv * pow<3>(one_over_rho);
  // dy_deta
  dX_dlogical[1] =
      cap_eta_deriv *
      ((scaled_trapezoid_zero_ + scaled_trapezoid_rate_ * xi) +
       (annulus_zero_ + annulus_rate_ * xi) * pow<3>(one_over_rho));
  dX_dlogical =
      discrete_rotation(orientation_of_wedge_, std::move(dX_dlogical));
  get<0, 1>(jacobian_matrix) = dX_dlogical[0];
  get<1, 1>(jacobian_matrix) = dX_dlogical[1];
  return jacobian_matrix;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame> Wedge2D::inv_jacobian(
    const std::array<T, 2>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& xi = source_coords[0];
  const ReturnType& eta = source_coords[1];
  const ReturnType& cap_eta = with_equiangular_map_ ? tan(M_PI_4 * eta) : eta;
  const ReturnType cap_eta_deriv = with_equiangular_map_
                                       ? M_PI_4 * (1.0 + square(cap_eta))
                                       : make_with_value<ReturnType>(eta, 1.0);
  const ReturnType one_over_rho = 1.0 / sqrt(1.0 + square(cap_eta));
  const ReturnType one_over_physical_x =
      1.0 / (scaled_trapezoid_zero_ + scaled_trapezoid_rate_ * xi +
             (annulus_zero_ + annulus_rate_ * xi) * one_over_rho);
  const ReturnType scaled_x_trapezoid =
      scaled_trapezoid_zero_ + scaled_trapezoid_rate_ * xi;
  const ReturnType annulus_factor_over_rho_cubed =
      (annulus_zero_ + annulus_rate_ * xi) * pow<3>(one_over_rho);
  const ReturnType one_over_dx_dxi =
      1.0 / (scaled_trapezoid_rate_ + annulus_rate_ * one_over_rho);
  const ReturnType dxi_factor = one_over_physical_x * one_over_dx_dxi;
  auto inv_jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);

  ReturnType dxi_dx =
      dxi_factor * (annulus_factor_over_rho_cubed + scaled_x_trapezoid);
  ReturnType dxi_dy = dxi_factor * annulus_factor_over_rho_cubed * cap_eta;
  std::array<ReturnType, 2> dlogical_dX = discrete_rotation(
      orientation_of_wedge_,
      std::array<ReturnType, 2>{{std::move(dxi_dx), std::move(dxi_dy)}});
  get<0, 0>(inv_jacobian_matrix) = dlogical_dX[0];
  get<0, 1>(inv_jacobian_matrix) = dlogical_dX[1];

  // deta_dy
  dlogical_dX[1] = one_over_physical_x / cap_eta_deriv;
  // deta_dx
  dlogical_dX[0] = -cap_eta * dlogical_dX[1];
  dlogical_dX =
      discrete_rotation(orientation_of_wedge_, std::move(dlogical_dX));
  get<1, 0>(inv_jacobian_matrix) = dlogical_dX[0];
  get<1, 1>(inv_jacobian_matrix) = dlogical_dX[1];
  return inv_jacobian_matrix;
}

void Wedge2D::pup(PUP::er& p) {
  p | radius_inner_;
  p | radius_outer_;
  p | circularity_inner_;
  p | circularity_outer_;
  p | orientation_of_wedge_;
  p | with_equiangular_map_;
  p | scaled_trapezoid_zero_;
  p | annulus_zero_;
  p | scaled_trapezoid_rate_;
  p | annulus_rate_;
}

bool operator==(const Wedge2D& lhs, const Wedge2D& rhs) noexcept {
  return lhs.radius_inner_ == rhs.radius_inner_ and
         lhs.radius_outer_ == rhs.radius_outer_ and
         lhs.circularity_inner_ == rhs.circularity_inner_ and
         lhs.circularity_outer_ == rhs.circularity_outer_ and
         lhs.orientation_of_wedge_ == rhs.orientation_of_wedge_ and
         lhs.with_equiangular_map_ == rhs.with_equiangular_map_ and
         lhs.scaled_trapezoid_zero_ == rhs.scaled_trapezoid_zero_ and
         lhs.annulus_zero_ == rhs.annulus_zero_ and
         lhs.scaled_trapezoid_rate_ == rhs.scaled_trapezoid_rate_ and
         lhs.annulus_rate_ == rhs.annulus_rate_;
}

bool operator!=(const Wedge2D& lhs, const Wedge2D& rhs) noexcept {
  return not(lhs == rhs);
}

// Explicit instantiations
/// \cond
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 2> Wedge2D::      \
  operator()(const std::array<DTYPE(data), 2>& source_coords) const noexcept; \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 2, Frame::NoFrame>  \
  Wedge2D::jacobian(const std::array<DTYPE(data), 2>& source_coords)          \
      const noexcept;                                                         \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 2, Frame::NoFrame>  \
  Wedge2D::inv_jacobian(const std::array<DTYPE(data), 2>& source_coords)      \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))
#undef DTYPE
#undef INSTANTIATE
/// \endcond
}  // namespace CoordinateMaps
}  // namespace domain
