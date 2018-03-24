// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Frustum.hpp"

#include <pup.h>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/OrientationMap.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace CoordinateMaps {
Frustum::Frustum(const std::array<std::array<double, 2>, 4>& face_vertices,
                 const double lower_bound, const double upper_bound,
                 OrientationMap<3> orientation_of_frustum) noexcept
    // clang-tidy: trivially copyable
    : orientation_of_frustum_(std::move(orientation_of_frustum)) {  // NOLINT
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
  sum_midpoint_x_ = 0.5 * (lower_x_upper_base + upper_x_upper_base +
                           lower_x_lower_base + upper_x_lower_base);
  dif_midpoint_x_ = 0.5 * (lower_x_upper_base + upper_x_upper_base -
                           lower_x_lower_base - upper_x_lower_base);
  sum_half_length_x_ = 0.5 * (upper_x_upper_base - lower_x_upper_base +
                              upper_x_lower_base - lower_x_lower_base);
  dif_half_length_x_ = 0.5 * (upper_x_upper_base - lower_x_upper_base -
                              upper_x_lower_base + lower_x_lower_base);
  sum_midpoint_y_ = 0.5 * (lower_y_upper_base + upper_y_upper_base +
                           lower_y_lower_base + upper_y_lower_base);
  dif_midpoint_y_ = 0.5 * (lower_y_upper_base + upper_y_upper_base -
                           lower_y_lower_base - upper_y_lower_base);
  sum_half_length_y_ = 0.5 * (upper_y_upper_base - lower_y_upper_base +
                              upper_y_lower_base - lower_y_lower_base);
  dif_half_length_y_ = 0.5 * (upper_y_upper_base - lower_y_upper_base -
                              upper_y_lower_base + lower_y_lower_base);
  midpoint_z_ = 0.5 * (upper_bound + lower_bound);
  half_length_z_ = 0.5 * (upper_bound - lower_bound);
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> Frustum::operator()(
    const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  const ReturnType& xi = source_coords[0];
  const ReturnType& eta = source_coords[1];
  const ReturnType& zeta = source_coords[2];

  ReturnType physical_x =
      0.5 * ((sum_midpoint_x_ + sum_half_length_x_ * xi) +
             (dif_midpoint_x_ + dif_half_length_x_ * xi) * zeta);
  ReturnType physical_y =
      0.5 * ((sum_midpoint_y_ + sum_half_length_y_ * eta) +
             (dif_midpoint_y_ + dif_half_length_y_ * eta) * zeta);
  ReturnType physical_z = midpoint_z_ + half_length_z_ * zeta;

  std::array<ReturnType, 3> physical_coords{
      {std::move(physical_x), std::move(physical_y), std::move(physical_z)}};
  return discrete_rotation(orientation_of_frustum_, std::move(physical_coords));
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> Frustum::inverse(
    const std::array<T, 3>& target_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  // physical coords {x,y,z}
  std::array<ReturnType, 3> coords =
      discrete_rotation(orientation_of_frustum_.inverse_map(), target_coords);

  // We reuse `coords` to avoid an allocation. This is now the
  // logical coords {xi,eta,zeta}
  coords[2] = (coords[2] - midpoint_z_) / half_length_z_;
  coords[0] =
      (2.0 * coords[0] - sum_midpoint_x_ - dif_midpoint_x_ * coords[2]) /
      (sum_half_length_x_ + dif_half_length_x_ * coords[2]);
  coords[1] =
      (2.0 * coords[1] - sum_midpoint_y_ - dif_midpoint_y_ * coords[2]) /
      (sum_half_length_y_ + dif_half_length_y_ * coords[2]);
  return coords;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> Frustum::jacobian(
    const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& xi = source_coords[0];
  const ReturnType& eta = source_coords[1];
  const ReturnType& zeta = source_coords[2];
  const auto zero = make_with_value<ReturnType>(xi, 0.0);

  auto jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);

  // dX_dxi
  std::array<ReturnType, 3> dX_dlogical = discrete_rotation(
      orientation_of_frustum_,
      std::array<ReturnType, 3>{
          {0.5 * (sum_half_length_x_ + dif_half_length_x_ * zeta), zero,
           zero}});
  get<0, 0>(jacobian_matrix) = dX_dlogical[0];
  get<1, 0>(jacobian_matrix) = dX_dlogical[1];
  get<2, 0>(jacobian_matrix) = dX_dlogical[2];

  // dX_deta
  dX_dlogical = discrete_rotation(
      orientation_of_frustum_,
      std::array<ReturnType, 3>{
          {zero, 0.5 * (sum_half_length_y_ + dif_half_length_y_ * zeta),
           zero}});
  get<0, 1>(jacobian_matrix) = dX_dlogical[0];
  get<1, 1>(jacobian_matrix) = dX_dlogical[1];
  get<2, 1>(jacobian_matrix) = dX_dlogical[2];

  // dX_dzeta
  dX_dlogical[0] = 0.5 * (dif_midpoint_x_ + dif_half_length_x_ * xi);
  dX_dlogical[1] = 0.5 * (dif_midpoint_y_ + dif_half_length_y_ * eta);
  dX_dlogical[2] = make_with_value<ReturnType>(xi, half_length_z_);
  dX_dlogical =
      discrete_rotation(orientation_of_frustum_, std::move(dX_dlogical));
  get<0, 2>(jacobian_matrix) = dX_dlogical[0];
  get<1, 2>(jacobian_matrix) = dX_dlogical[1];
  get<2, 2>(jacobian_matrix) = dX_dlogical[2];
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
  p | sum_midpoint_x_;
  p | dif_midpoint_x_;
  p | sum_half_length_x_;
  p | dif_half_length_x_;
  p | sum_midpoint_y_;
  p | dif_midpoint_y_;
  p | sum_half_length_y_;
  p | dif_half_length_y_;
  p | midpoint_z_;
  p | half_length_z_;
}

bool operator==(const Frustum& lhs, const Frustum& rhs) noexcept {
  return lhs.orientation_of_frustum_ == rhs.orientation_of_frustum_ and
         lhs.sum_midpoint_x_ == rhs.sum_midpoint_x_ and
         lhs.dif_midpoint_x_ == rhs.dif_midpoint_x_ and
         lhs.sum_half_length_x_ == rhs.sum_half_length_x_ and
         lhs.dif_half_length_x_ == rhs.dif_half_length_x_ and
         lhs.sum_midpoint_y_ == rhs.sum_midpoint_y_ and
         lhs.dif_midpoint_y_ == rhs.dif_midpoint_y_ and
         lhs.sum_half_length_y_ == lhs.sum_half_length_y_ and
         lhs.dif_half_length_y_ == lhs.dif_half_length_y_ and
         lhs.midpoint_z_ == rhs.midpoint_z_ and
         lhs.half_length_z_ == rhs.half_length_z_;
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
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>                \
  Frustum::inverse(const std::array<DTYPE(data), 3>& target_coords)           \
      const noexcept;                                                         \
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
