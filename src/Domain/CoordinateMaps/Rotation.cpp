// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Rotation.hpp"

#include "DataStructures/MakeWithValue.hpp"
#include "Utilities/DereferenceWrapper.hpp"

namespace CoordinateMaps {

Rotation<2>::Rotation(const double rotation_angle)
    : rotation_angle_(rotation_angle),
      rotation_matrix_(std::numeric_limits<double>::signaling_NaN()) {
  const double cos_alpha = cos(rotation_angle_);
  const double sin_alpha = sin(rotation_angle_);
  get<0, 0>(rotation_matrix_) = cos_alpha;
  get<0, 1>(rotation_matrix_) = -sin_alpha;
  get<1, 0>(rotation_matrix_) = sin_alpha;
  get<1, 1>(rotation_matrix_) = cos_alpha;
}

template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 2> Rotation<2>::
operator()(const std::array<T, 2>& source_coords) const {
  return {{source_coords[0] * get<0, 0>(rotation_matrix_) +
               source_coords[1] * get<0, 1>(rotation_matrix_),
           source_coords[0] * get<1, 0>(rotation_matrix_) +
               source_coords[1] * get<1, 1>(rotation_matrix_)}};
}

template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 2>
Rotation<2>::inverse(const std::array<T, 2>& target_coords) const {
  return {{target_coords[0] * get<0, 0>(rotation_matrix_) +
               target_coords[1] * get<1, 0>(rotation_matrix_),
           target_coords[0] * get<0, 1>(rotation_matrix_) +
               target_coords[1] * get<1, 1>(rotation_matrix_)}};
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Rotation<2>::jacobian(const std::array<T, 2>& source_coords) const {
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
      jac{make_with_value<std::decay_t<tt::remove_reference_wrapper_t<T>>>(
          dereference_wrapper(source_coords[0]), 0.0)};
  get<0, 0>(jac) = get<0, 0>(rotation_matrix_);
  get<1, 0>(jac) = get<1, 0>(rotation_matrix_);
  get<0, 1>(jac) = get<0, 1>(rotation_matrix_);
  get<1, 1>(jac) = get<1, 1>(rotation_matrix_);
  return jac;
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Rotation<2>::inv_jacobian(const std::array<T, 2>& source_coords) const {
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
      inv_jac{make_with_value<std::decay_t<tt::remove_reference_wrapper_t<T>>>(
          dereference_wrapper(source_coords[0]), 0.0)};
  get<0, 0>(inv_jac) = get<0, 0>(rotation_matrix_);
  get<1, 0>(inv_jac) = get<0, 1>(rotation_matrix_);
  get<0, 1>(inv_jac) = get<1, 0>(rotation_matrix_);
  get<1, 1>(inv_jac) = get<1, 1>(rotation_matrix_);
  return inv_jac;
}

void Rotation<2>::pup(PUP::er& p) {
  p | rotation_angle_;
  p | rotation_matrix_;
}

bool operator==(const Rotation<2>& lhs, const Rotation<2>& rhs) noexcept {
  return lhs.rotation_angle_ == rhs.rotation_angle_;
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
      rotation_matrix_(std::numeric_limits<double>::signaling_NaN()) {
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
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 3> Rotation<3>::
operator()(const std::array<T, 3>& source_coords) const {
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

template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 3>
Rotation<3>::inverse(const std::array<T, 3>& target_coords) const {
  // Inverse rotation matrix is the same as the transpose.
  return {{target_coords[0] * get<0, 0>(rotation_matrix_) +
               target_coords[1] * get<1, 0>(rotation_matrix_) +
               target_coords[2] * get<2, 0>(rotation_matrix_),
           target_coords[0] * get<0, 1>(rotation_matrix_) +
               target_coords[1] * get<1, 1>(rotation_matrix_) +
               target_coords[2] * get<2, 1>(rotation_matrix_),
           target_coords[0] * get<0, 2>(rotation_matrix_) +
               target_coords[1] * get<1, 2>(rotation_matrix_) +
               target_coords[2] * get<2, 2>(rotation_matrix_)}};
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Rotation<3>::jacobian(const std::array<T, 3>& source_coords) const {
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
      jac{make_with_value<std::decay_t<tt::remove_reference_wrapper_t<T>>>(
          dereference_wrapper(source_coords[0]), 0.0)};
  get<0, 0>(jac) = get<0, 0>(rotation_matrix_);
  get<1, 0>(jac) = get<1, 0>(rotation_matrix_);
  get<0, 1>(jac) = get<0, 1>(rotation_matrix_);
  get<1, 1>(jac) = get<1, 1>(rotation_matrix_);
  get<2, 0>(jac) = get<2, 0>(rotation_matrix_);
  get<2, 1>(jac) = get<2, 1>(rotation_matrix_);
  get<0, 2>(jac) = get<0, 2>(rotation_matrix_);
  get<1, 2>(jac) = get<1, 2>(rotation_matrix_);
  get<2, 2>(jac) = get<2, 2>(rotation_matrix_);
  return jac;
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Rotation<3>::inv_jacobian(const std::array<T, 3>& source_coords) const {
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
      inv_jac{make_with_value<std::decay_t<tt::remove_reference_wrapper_t<T>>>(
          dereference_wrapper(source_coords[0]), 0.0)};
  get<0, 0>(inv_jac) = get<0, 0>(rotation_matrix_);
  get<1, 0>(inv_jac) = get<0, 1>(rotation_matrix_);
  get<0, 1>(inv_jac) = get<1, 0>(rotation_matrix_);
  get<1, 1>(inv_jac) = get<1, 1>(rotation_matrix_);
  get<2, 0>(inv_jac) = get<0, 2>(rotation_matrix_);
  get<2, 1>(inv_jac) = get<1, 2>(rotation_matrix_);
  get<0, 2>(inv_jac) = get<2, 0>(rotation_matrix_);
  get<1, 2>(inv_jac) = get<2, 1>(rotation_matrix_);
  get<2, 2>(inv_jac) = get<2, 2>(rotation_matrix_);
  return inv_jac;
}

void Rotation<3>::pup(PUP::er& p) {  // NOLINT
  p | rotation_about_z_;
  p | rotation_about_rotated_y_;
  p | rotation_about_rotated_z_;
  p | rotation_matrix_;
}

bool operator==(const Rotation<3>& lhs, const Rotation<3>& rhs) noexcept {
  return lhs.rotation_about_z_ == rhs.rotation_about_z_ and
         lhs.rotation_about_rotated_y_ == rhs.rotation_about_rotated_y_ and
         lhs.rotation_about_rotated_z_ == rhs.rotation_about_rotated_z_;
}

bool operator!=(const Rotation<3>& lhs, const Rotation<3>& rhs) noexcept {
  return not(lhs == rhs);
}

template std::array<double, 2> Rotation<2>::operator()(
    const std::array<std::reference_wrapper<const double>, 2>& source_coords)
    const;
template std::array<double, 2> Rotation<2>::operator()(
    const std::array<double, 2>& source_coords) const;
template std::array<DataVector, 2> Rotation<2>::operator()(
    const std::array<std::reference_wrapper<const DataVector>, 2>&
        source_coords) const;
template std::array<DataVector, 2> Rotation<2>::operator()(
    const std::array<DataVector, 2>& source_coords) const;

template std::array<double, 2> Rotation<2>::inverse(
    const std::array<std::reference_wrapper<const double>, 2>& target_coords)
    const;
template std::array<double, 2> Rotation<2>::inverse(
    const std::array<double, 2>& target_coords) const;
template std::array<DataVector, 2> Rotation<2>::inverse(
    const std::array<std::reference_wrapper<const DataVector>, 2>&
        target_coords) const;
template std::array<DataVector, 2> Rotation<2>::inverse(
    const std::array<DataVector, 2>& target_coords) const;

template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Rotation<2>::jacobian(const std::array<std::reference_wrapper<const double>, 2>&
                          source_coords) const;
template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Rotation<2>::jacobian(const std::array<double, 2>& source_coords) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Rotation<2>::jacobian(const std::array<std::reference_wrapper<const DataVector>,
                                       2>& source_coords) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Rotation<2>::jacobian(const std::array<DataVector, 2>& source_coords) const;

template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Rotation<2>::inv_jacobian(const std::array<std::reference_wrapper<const double>,
                                           2>& source_coords) const;
template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Rotation<2>::inv_jacobian(const std::array<double, 2>& source_coords) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Rotation<2>::inv_jacobian(
    const std::array<std::reference_wrapper<const DataVector>, 2>&
        source_coords) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Rotation<2>::inv_jacobian(const std::array<DataVector, 2>& source_coords) const;

template std::array<double, 3> Rotation<3>::operator()(
    const std::array<std::reference_wrapper<const double>, 3>& source_coords)
    const;
template std::array<double, 3> Rotation<3>::operator()(
    const std::array<double, 3>& source_coords) const;
template std::array<DataVector, 3> Rotation<3>::operator()(
    const std::array<std::reference_wrapper<const DataVector>, 3>&
        source_coords) const;
template std::array<DataVector, 3> Rotation<3>::operator()(
    const std::array<DataVector, 3>& source_coords) const;

template std::array<double, 3> Rotation<3>::inverse(
    const std::array<std::reference_wrapper<const double>, 3>& target_coords)
    const;
template std::array<double, 3> Rotation<3>::inverse(
    const std::array<double, 3>& target_coords) const;
template std::array<DataVector, 3> Rotation<3>::inverse(
    const std::array<std::reference_wrapper<const DataVector>, 3>&
        target_coords) const;
template std::array<DataVector, 3> Rotation<3>::inverse(
    const std::array<DataVector, 3>& target_coords) const;

template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Rotation<3>::jacobian(const std::array<std::reference_wrapper<const double>, 3>&
                          source_coords) const;
template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Rotation<3>::jacobian(const std::array<double, 3>& source_coords) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Rotation<3>::jacobian(const std::array<std::reference_wrapper<const DataVector>,
                                       3>& source_coords) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Rotation<3>::jacobian(const std::array<DataVector, 3>& source_coords) const;

template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Rotation<3>::inv_jacobian(const std::array<std::reference_wrapper<const double>,
                                           3>& source_coords) const;
template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Rotation<3>::inv_jacobian(const std::array<double, 3>& source_coords) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Rotation<3>::inv_jacobian(
    const std::array<std::reference_wrapper<const DataVector>, 3>&
        source_coords) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Rotation<3>::inv_jacobian(const std::array<DataVector, 3>& source_coords) const;

}  // namespace CoordinateMaps
