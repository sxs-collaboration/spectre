// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Rotation.hpp"

#include <cmath>
#include <pup.h>

#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain {
namespace CoordinateMaps {

Rotation<2>::Rotation(const double rotation_angle)
    : rotation_angle_(rotation_angle),
      rotation_matrix_(std::numeric_limits<double>::signaling_NaN()),
      is_identity_(rotation_angle_ == 0.0) {
  const double cos_alpha = cos(rotation_angle_);
  const double sin_alpha = sin(rotation_angle_);
  get<0, 0>(rotation_matrix_) = cos_alpha;
  get<0, 1>(rotation_matrix_) = -sin_alpha;
  get<1, 0>(rotation_matrix_) = sin_alpha;
  get<1, 1>(rotation_matrix_) = cos_alpha;
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 2> Rotation<2>::operator()(
    const std::array<T, 2>& source_coords) const noexcept {
  return {{source_coords[0] * get<0, 0>(rotation_matrix_) +
               source_coords[1] * get<0, 1>(rotation_matrix_),
           source_coords[0] * get<1, 0>(rotation_matrix_) +
               source_coords[1] * get<1, 1>(rotation_matrix_)}};
}

boost::optional<std::array<double, 2>> Rotation<2>::inverse(
    const std::array<double, 2>& target_coords) const noexcept {
  return {{{target_coords[0] * get<0, 0>(rotation_matrix_) +
                target_coords[1] * get<1, 0>(rotation_matrix_),
            target_coords[0] * get<0, 1>(rotation_matrix_) +
                target_coords[1] * get<1, 1>(rotation_matrix_)}}};
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame> Rotation<2>::jacobian(
    const std::array<T, 2>& source_coords) const noexcept {
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame> jacobian_matrix{
      make_with_value<tt::remove_cvref_wrap_t<T>>(
          dereference_wrapper(source_coords[0]), 0.0)};
  get<0, 0>(jacobian_matrix) = get<0, 0>(rotation_matrix_);
  get<1, 0>(jacobian_matrix) = get<1, 0>(rotation_matrix_);
  get<0, 1>(jacobian_matrix) = get<0, 1>(rotation_matrix_);
  get<1, 1>(jacobian_matrix) = get<1, 1>(rotation_matrix_);
  return jacobian_matrix;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame>
Rotation<2>::inv_jacobian(const std::array<T, 2>& source_coords) const
    noexcept {
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame> inv_jacobian_matrix{
      make_with_value<tt::remove_cvref_wrap_t<T>>(
          dereference_wrapper(source_coords[0]), 0.0)};
  get<0, 0>(inv_jacobian_matrix) = get<0, 0>(rotation_matrix_);
  get<1, 0>(inv_jacobian_matrix) = get<0, 1>(rotation_matrix_);
  get<0, 1>(inv_jacobian_matrix) = get<1, 0>(rotation_matrix_);
  get<1, 1>(inv_jacobian_matrix) = get<1, 1>(rotation_matrix_);
  return inv_jacobian_matrix;
}

void Rotation<2>::pup(PUP::er& p) {
  p | rotation_angle_;
  p | rotation_matrix_;
  p | is_identity_;
}

bool operator==(const Rotation<2>& lhs, const Rotation<2>& rhs) noexcept {
  return lhs.rotation_angle_ == rhs.rotation_angle_ and
         lhs.is_identity_ == rhs.is_identity_;
}

bool operator!=(const Rotation<2>& lhs, const Rotation<2>& rhs) noexcept {
  return not(lhs == rhs);
}

Rotation<3>::Rotation(const double rotation_about_z,
                      const double rotation_about_rotated_y,
                      const double rotation_about_rotated_z)
    : rotation_about_z_(rotation_about_z),
      rotation_about_rotated_y_(rotation_about_rotated_y),
      rotation_about_rotated_z_(rotation_about_rotated_z),
      rotation_matrix_(std::numeric_limits<double>::signaling_NaN()),
      is_identity_(rotation_about_z_ == 0.0 and
                   rotation_about_rotated_y_ == 0.0 and
                   rotation_about_rotated_z_ == 0.0) {
  const double cos_alpha = cos(rotation_about_z_);
  const double sin_alpha = sin(rotation_about_z_);
  const double cos_beta = cos(rotation_about_rotated_y_);
  const double sin_beta = sin(rotation_about_rotated_y_);
  const double cos_gamma = cos(rotation_about_rotated_z_);
  const double sin_gamma = sin(rotation_about_rotated_z_);
  get<0, 0>(rotation_matrix_) =
      cos_gamma * cos_beta * cos_alpha - sin_gamma * sin_alpha;
  get<0, 1>(rotation_matrix_) =
      -sin_gamma * cos_beta * cos_alpha - cos_gamma * sin_alpha;
  get<0, 2>(rotation_matrix_) = sin_beta * cos_alpha;
  get<1, 0>(rotation_matrix_) =
      cos_gamma * cos_beta * sin_alpha + sin_gamma * cos_alpha;
  get<1, 1>(rotation_matrix_) =
      -sin_gamma * cos_beta * sin_alpha + cos_gamma * cos_alpha;
  get<1, 2>(rotation_matrix_) = sin_beta * sin_alpha;
  get<2, 0>(rotation_matrix_) = -cos_gamma * sin_beta;
  get<2, 1>(rotation_matrix_) = sin_gamma * sin_beta;
  get<2, 2>(rotation_matrix_) = cos_beta;
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> Rotation<3>::operator()(
    const std::array<T, 3>& source_coords) const noexcept {
  return {{source_coords[0] * get<0, 0>(rotation_matrix_) +
               source_coords[1] * get<0, 1>(rotation_matrix_) +
               source_coords[2] * get<0, 2>(rotation_matrix_),
           source_coords[0] * get<1, 0>(rotation_matrix_) +
               source_coords[1] * get<1, 1>(rotation_matrix_) +
               source_coords[2] * get<1, 2>(rotation_matrix_),
           source_coords[0] * get<2, 0>(rotation_matrix_) +
               source_coords[1] * get<2, 1>(rotation_matrix_) +
               source_coords[2] * get<2, 2>(rotation_matrix_)}};
}

boost::optional<std::array<double, 3>> Rotation<3>::inverse(
    const std::array<double, 3>& target_coords) const noexcept {
  // Inverse rotation matrix is the same as the transpose.
  return {{{target_coords[0] * get<0, 0>(rotation_matrix_) +
                target_coords[1] * get<1, 0>(rotation_matrix_) +
                target_coords[2] * get<2, 0>(rotation_matrix_),
            target_coords[0] * get<0, 1>(rotation_matrix_) +
                target_coords[1] * get<1, 1>(rotation_matrix_) +
                target_coords[2] * get<2, 1>(rotation_matrix_),
            target_coords[0] * get<0, 2>(rotation_matrix_) +
                target_coords[1] * get<1, 2>(rotation_matrix_) +
                target_coords[2] * get<2, 2>(rotation_matrix_)}}};
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> Rotation<3>::jacobian(
    const std::array<T, 3>& source_coords) const noexcept {
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian_matrix{
      make_with_value<tt::remove_cvref_wrap_t<T>>(
          dereference_wrapper(source_coords[0]), 0.0)};
  get<0, 0>(jacobian_matrix) = get<0, 0>(rotation_matrix_);
  get<1, 0>(jacobian_matrix) = get<1, 0>(rotation_matrix_);
  get<0, 1>(jacobian_matrix) = get<0, 1>(rotation_matrix_);
  get<1, 1>(jacobian_matrix) = get<1, 1>(rotation_matrix_);
  get<2, 0>(jacobian_matrix) = get<2, 0>(rotation_matrix_);
  get<2, 1>(jacobian_matrix) = get<2, 1>(rotation_matrix_);
  get<0, 2>(jacobian_matrix) = get<0, 2>(rotation_matrix_);
  get<1, 2>(jacobian_matrix) = get<1, 2>(rotation_matrix_);
  get<2, 2>(jacobian_matrix) = get<2, 2>(rotation_matrix_);
  return jacobian_matrix;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
Rotation<3>::inv_jacobian(const std::array<T, 3>& source_coords) const
    noexcept {
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian_matrix{
      make_with_value<tt::remove_cvref_wrap_t<T>>(
          dereference_wrapper(source_coords[0]), 0.0)};
  get<0, 0>(inv_jacobian_matrix) = get<0, 0>(rotation_matrix_);
  get<1, 0>(inv_jacobian_matrix) = get<0, 1>(rotation_matrix_);
  get<0, 1>(inv_jacobian_matrix) = get<1, 0>(rotation_matrix_);
  get<1, 1>(inv_jacobian_matrix) = get<1, 1>(rotation_matrix_);
  get<2, 0>(inv_jacobian_matrix) = get<0, 2>(rotation_matrix_);
  get<2, 1>(inv_jacobian_matrix) = get<1, 2>(rotation_matrix_);
  get<0, 2>(inv_jacobian_matrix) = get<2, 0>(rotation_matrix_);
  get<1, 2>(inv_jacobian_matrix) = get<2, 1>(rotation_matrix_);
  get<2, 2>(inv_jacobian_matrix) = get<2, 2>(rotation_matrix_);
  return inv_jacobian_matrix;
}

void Rotation<3>::pup(PUP::er& p) {  // NOLINT
  p | rotation_about_z_;
  p | rotation_about_rotated_y_;
  p | rotation_about_rotated_z_;
  p | rotation_matrix_;
  p | is_identity_;
}

bool operator==(const Rotation<3>& lhs, const Rotation<3>& rhs) noexcept {
  return lhs.rotation_about_z_ == rhs.rotation_about_z_ and
         lhs.rotation_about_rotated_y_ == rhs.rotation_about_rotated_y_ and
         lhs.rotation_about_rotated_z_ == rhs.rotation_about_rotated_z_ and
         lhs.is_identity_ == rhs.is_identity_;
}

bool operator!=(const Rotation<3>& lhs, const Rotation<3>& rhs) noexcept {
  return not(lhs == rhs);
}
// Explicit instantiations
/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                   \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data)>         \
  Rotation<DIM(data)>::operator()(                                             \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const noexcept; \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),           \
                    Frame::NoFrame>                                            \
  Rotation<DIM(data)>::jacobian(                                               \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const noexcept; \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),           \
                    Frame::NoFrame>                                            \
  Rotation<DIM(data)>::inv_jacobian(                                           \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3),
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))
#undef DIM
#undef DTYPE
#undef INSTANTIATE
/// \endcond

}  // namespace CoordinateMaps
}  // namespace domain
