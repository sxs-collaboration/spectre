// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Frustum.hpp"

#include <algorithm>
#include <boost/none.hpp>
#include <cmath>
#include <pup.h>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/Side.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain {
namespace CoordinateMaps {

Frustum::Frustum(const std::array<std::array<double, 2>, 4>& face_vertices,
                 const double lower_bound, const double upper_bound,
                 OrientationMap<3> orientation_of_frustum,
                 const bool with_equiangular_map,
                 const double projective_scale_factor,
                 const bool auto_projective_scale_factor) noexcept
    // clang-tidy: trivially copyable
    : orientation_of_frustum_(std::move(orientation_of_frustum)),  // NOLINT
      with_equiangular_map_(with_equiangular_map),
      is_identity_(face_vertices ==
                       std::array<std::array<double, 2>, 4>{{{{-1.0, -1.0}},
                                                             {{1.0, 1.0}},
                                                             {{-1.0, -1.0}},
                                                             {{1.0, 1.0}}}} and
                   lower_bound == -1.0 and upper_bound == 1.0 and
                   orientation_of_frustum_ == OrientationMap<3>{} and
                   not with_equiangular_map and projective_scale_factor == 1.0),
      with_projective_map_(projective_scale_factor != 1.0) {
  ASSERT(projective_scale_factor != 0.0,
         "A projective scale factor of zero maps all coordinates to zero! Set "
         "projective_scale_factor to unity to turn off projective scaling.");
  const double& lower_x_lower_base = face_vertices[0][0];
  const double& lower_y_lower_base = face_vertices[0][1];
  const double& upper_x_lower_base = face_vertices[1][0];
  const double& upper_y_lower_base = face_vertices[1][1];
  const double& lower_x_upper_base = face_vertices[2][0];
  const double& lower_y_upper_base = face_vertices[2][1];
  const double& upper_x_upper_base = face_vertices[3][0];
  const double& upper_y_upper_base = face_vertices[3][1];
  ASSERT(upper_x_lower_base > lower_x_lower_base,
         "The lower bound for a coordinate must be numerically less"
         " than the upper bound for that coordinate.");
  ASSERT(upper_y_lower_base > lower_y_lower_base,
         "The lower bound for a coordinate must be numerically less"
         " than the upper bound for that coordinate.");
  ASSERT(upper_x_upper_base > lower_x_upper_base,
         "The lower bound for a coordinate must be numerically less"
         " than the upper bound for that coordinate.");
  ASSERT(upper_y_upper_base > lower_y_upper_base,
         "The lower bound for a coordinate must be numerically less"
         " than the upper bound for that coordinate.");
  ASSERT(upper_bound > lower_bound,
         "The lower bound for a coordinate must be numerically less"
         " than the upper bound for that coordinate.");
  sigma_x_ = 0.25 * (lower_x_upper_base + upper_x_upper_base +
                     lower_x_lower_base + upper_x_lower_base);
  delta_x_zeta_ = 0.25 * (lower_x_upper_base + upper_x_upper_base -
                          lower_x_lower_base - upper_x_lower_base);
  delta_x_xi_ = 0.25 * (upper_x_upper_base - lower_x_upper_base +
                        upper_x_lower_base - lower_x_lower_base);
  delta_x_xi_zeta_ = 0.25 * (upper_x_upper_base - lower_x_upper_base -
                             upper_x_lower_base + lower_x_lower_base);
  sigma_y_ = 0.25 * (lower_y_upper_base + upper_y_upper_base +
                     lower_y_lower_base + upper_y_lower_base);
  delta_y_zeta_ = 0.25 * (lower_y_upper_base + upper_y_upper_base -
                          lower_y_lower_base - upper_y_lower_base);
  delta_y_eta_ = 0.25 * (upper_y_upper_base - lower_y_upper_base +
                         upper_y_lower_base - lower_y_lower_base);
  delta_y_eta_zeta_ = 0.25 * (upper_y_upper_base - lower_y_upper_base -
                              upper_y_lower_base + lower_y_lower_base);
  sigma_z_ = 0.5 * (upper_bound + lower_bound);
  delta_z_zeta_ = 0.5 * (upper_bound - lower_bound);
  w_plus_ = projective_scale_factor + 1.0;
  w_minus_ = projective_scale_factor - 1.0;
  if (auto_projective_scale_factor) {
    with_projective_map_ = true;
    const double w_delta = sqrt(((upper_x_lower_base - lower_x_lower_base) *
                                 (upper_y_lower_base - lower_y_lower_base)) /
                                ((upper_x_upper_base - lower_x_upper_base) *
                                 (upper_y_upper_base - lower_y_upper_base)));
    w_plus_ = w_delta + 1.0;
    w_minus_ = w_delta - 1.0;
  }
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> Frustum::operator()(
    const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  const ReturnType& xi = source_coords[0];
  const ReturnType& eta = source_coords[1];
  const ReturnType& zeta = source_coords[2];
  const ReturnType cap_zeta =
      with_projective_map_
          ? (w_minus_ + w_plus_ * zeta) / (w_plus_ + w_minus_ * zeta)
          : zeta;
  const ReturnType cap_xi = with_equiangular_map_ ? tan(M_PI_4 * xi) : xi;
  const ReturnType cap_eta = with_equiangular_map_ ? tan(M_PI_4 * eta) : eta;
  ReturnType physical_x =
      sigma_x_ + delta_x_xi_ * cap_xi +
      (delta_x_zeta_ + delta_x_xi_zeta_ * cap_xi) * cap_zeta;
  ReturnType physical_y =
      sigma_y_ + delta_y_eta_ * cap_eta +
      (delta_y_zeta_ + delta_y_eta_zeta_ * cap_eta) * cap_zeta;
  ReturnType physical_z = sigma_z_ + delta_z_zeta_ * cap_zeta;
  std::array<ReturnType, 3> physical_coords{
      {std::move(physical_x), std::move(physical_y), std::move(physical_z)}};
  return discrete_rotation(orientation_of_frustum_, std::move(physical_coords));
}

boost::optional<std::array<double, 3>> Frustum::inverse(
    const std::array<double, 3>& target_coords) const noexcept {
  // physical coords {x,y,z}
  std::array<double, 3> physical_coords =
      discrete_rotation(orientation_of_frustum_.inverse_map(), target_coords);

  // logical coords {xi,eta,zeta}
  std::array<double, 3> logical_coords{};

  logical_coords[2] = (physical_coords[2] - sigma_z_) / delta_z_zeta_;
  const auto denom0 = delta_x_xi_ + delta_x_xi_zeta_ * logical_coords[2];
  const auto denom1 = delta_y_eta_ + delta_y_eta_zeta_ * logical_coords[2];
  // denom0 and denom1 are always positive inside the frustum.
  if (denom0 < 0.0 or equal_within_roundoff(denom0, 0.0) or denom1 < 0.0 or
      equal_within_roundoff(denom1, 0.0)) {
    return boost::none;
  }

  logical_coords[0] =
      (physical_coords[0] - sigma_x_ - delta_x_zeta_ * logical_coords[2]) /
      denom0;
  logical_coords[1] =
      (physical_coords[1] - sigma_y_ - delta_y_zeta_ * logical_coords[2]) /
      denom1;
  if (with_equiangular_map_) {
    logical_coords[0] = atan(logical_coords[0]) / M_PI_4;
    logical_coords[1] = atan(logical_coords[1]) / M_PI_4;
  }
  if (with_projective_map_) {
    logical_coords[2] = (-w_minus_ + w_plus_ * logical_coords[2]) /
                        (w_plus_ - w_minus_ * logical_coords[2]);
  }
  return logical_coords;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> Frustum::jacobian(
    const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& xi = source_coords[0];
  const ReturnType& eta = source_coords[1];
  const ReturnType& zeta = source_coords[2];
  const ReturnType& cap_xi = with_equiangular_map_ ? tan(M_PI_4 * xi) : xi;
  const ReturnType& cap_eta = with_equiangular_map_ ? tan(M_PI_4 * eta) : eta;
  const ReturnType cap_zeta =
      with_projective_map_
          ? (w_minus_ + w_plus_ * zeta) / (w_plus_ + w_minus_ * zeta)
          : zeta;
  const ReturnType cap_xi_deriv = with_equiangular_map_
                                      ? M_PI_4 * (1.0 + square(cap_xi))
                                      : make_with_value<ReturnType>(xi, 1.0);

  const ReturnType cap_eta_deriv = with_equiangular_map_
                                       ? M_PI_4 * (1.0 + square(cap_eta))
                                       : make_with_value<ReturnType>(eta, 1.0);
  const ReturnType cap_zeta_deriv =
      with_projective_map_ ? (square(w_plus_) - square(w_minus_)) /
                                 square(w_plus_ + zeta * w_minus_)
                           : make_with_value<ReturnType>(zeta, 1.0);

  auto jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);

  // dX_dxi
  const auto mapped_xi =
      orientation_of_frustum_.inverse_map()(Direction<3>::upper_xi());
  const size_t mapped_dim_0 = orientation_of_frustum_.inverse_map()(0);
  jacobian_matrix.get(mapped_dim_0, 0) =
      delta_x_xi_ + delta_x_xi_zeta_ * cap_zeta;
  if (with_equiangular_map_) {
    jacobian_matrix.get(mapped_dim_0, 0) *= cap_xi_deriv;
  }
  if (mapped_xi.side() == Side::Lower) {
    jacobian_matrix.get(mapped_dim_0, 0) *= -1.0;
  }

  // dX_deta
  const auto mapped_eta =
      orientation_of_frustum_.inverse_map()(Direction<3>::upper_eta());
  const size_t mapped_dim_1 = orientation_of_frustum_.inverse_map()(1);
  jacobian_matrix.get(mapped_dim_1, 1) =
      delta_y_eta_ + delta_y_eta_zeta_ * cap_zeta;
  if (with_equiangular_map_) {
    jacobian_matrix.get(mapped_dim_1, 1) *= cap_eta_deriv;
  }
  if (mapped_eta.side() == Side::Lower) {
    jacobian_matrix.get(mapped_dim_1, 1) *= -1.0;
  }

  // dX_dzeta
  std::array<ReturnType, 3> dX_dzeta = discrete_rotation(
      orientation_of_frustum_,
      std::array<ReturnType, 3>{
          {delta_x_zeta_ + delta_x_xi_zeta_ * cap_xi,
           delta_y_zeta_ + delta_y_eta_zeta_ * cap_eta,
           make_with_value<ReturnType>(zeta, delta_z_zeta_)}});

  if (with_projective_map_) {
    dX_dzeta[0] *= cap_zeta_deriv;
    dX_dzeta[1] *= cap_zeta_deriv;
    dX_dzeta[2] *= cap_zeta_deriv;
  }

  get<0, 2>(jacobian_matrix) = dX_dzeta[0];
  get<1, 2>(jacobian_matrix) = dX_dzeta[1];
  get<2, 2>(jacobian_matrix) = dX_dzeta[2];

  return jacobian_matrix;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> Frustum::inv_jacobian(
    const std::array<T, 3>& source_coords) const noexcept {
  const auto jac = jacobian(source_coords);
  return determinant_and_inverse(jac).second;
}

void Frustum::pup(PUP::er& p) noexcept {
  p | orientation_of_frustum_;
  p | with_equiangular_map_;
  p | is_identity_;
  p | with_projective_map_;
  p | sigma_x_;
  p | delta_x_zeta_;
  p | delta_x_xi_;
  p | delta_x_xi_zeta_;
  p | sigma_y_;
  p | delta_y_zeta_;
  p | delta_y_eta_;
  p | delta_y_eta_zeta_;
  p | sigma_z_;
  p | delta_z_zeta_;
  p | w_plus_;
  p | w_minus_;
}

bool operator==(const Frustum& lhs, const Frustum& rhs) noexcept {
  return lhs.orientation_of_frustum_ == rhs.orientation_of_frustum_ and
         lhs.with_equiangular_map_ == rhs.with_equiangular_map_ and
         lhs.is_identity_ == rhs.is_identity_ and
         lhs.with_projective_map_ == rhs.with_projective_map_ and
         lhs.sigma_x_ == rhs.sigma_x_ and
         lhs.delta_x_zeta_ == rhs.delta_x_zeta_ and
         lhs.delta_x_xi_ == rhs.delta_x_xi_ and
         lhs.delta_x_xi_zeta_ == rhs.delta_x_xi_zeta_ and
         lhs.sigma_y_ == rhs.sigma_y_ and
         lhs.delta_y_zeta_ == rhs.delta_y_zeta_ and
         lhs.delta_y_eta_ == lhs.delta_y_eta_ and
         lhs.delta_y_eta_zeta_ == lhs.delta_y_eta_zeta_ and
         lhs.sigma_z_ == rhs.sigma_z_ and
         lhs.delta_z_zeta_ == rhs.delta_z_zeta_ and
         lhs.w_plus_ == rhs.w_plus_ and lhs.w_minus_ == rhs.w_minus_;
}

bool operator!=(const Frustum& lhs, const Frustum& rhs) noexcept {
  return not(lhs == rhs);
}

// Explicit instantiations
/// \cond
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3> Frustum::      \
  operator()(const std::array<DTYPE(data), 3>& source_coords) const noexcept; \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>  \
  Frustum::jacobian(const std::array<DTYPE(data), 3>& source_coords)          \
      const noexcept;                                                         \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>  \
  Frustum::inv_jacobian(const std::array<DTYPE(data), 3>& source_coords)      \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))
#undef DTYPE
#undef INSTANTIATE
/// \endcond
}  // namespace CoordinateMaps
}  // namespace domain
