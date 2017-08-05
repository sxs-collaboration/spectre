// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/EmbeddingMaps/Rotation.hpp"

namespace CoordinateMaps {

Rotation<2>::Rotation(const double rotation_angle)
    : rotation_angle_(rotation_angle),
      rotation_matrix_(std::numeric_limits<double>::signaling_NaN()) {
  const double cos_alpha = cos(rotation_angle_);
  const double sin_alpha = sin(rotation_angle_);
  rotation_matrix_.template get<0, 0>() = cos_alpha;
  rotation_matrix_.template get<0, 1>() = -sin_alpha;
  rotation_matrix_.template get<1, 0>() = sin_alpha;
  rotation_matrix_.template get<1, 1>() = cos_alpha;
}

template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 2> Rotation<2>::
operator()(const std::array<T, 2>& xi) const {
  return {{xi[0] * rotation_matrix_.template get<0, 0>() +
               xi[1] * rotation_matrix_.template get<0, 1>(),
           xi[0] * rotation_matrix_.template get<1, 0>() +
               xi[1] * rotation_matrix_.template get<1, 1>()}};
}

template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 2>
Rotation<2>::inverse(const std::array<T, 2>& x) const {
  return {{x[0] * rotation_matrix_.template get<0, 0>() +
               x[1] * rotation_matrix_.template get<1, 0>(),
           x[0] * rotation_matrix_.template get<0, 1>() +
               x[1] * rotation_matrix_.template get<1, 1>()}};
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Rotation<2>::jacobian(const std::array<T, 2>& /*xi*/) const {
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
      jac{};
  jac.template get<0, 0>() = rotation_matrix_.template get<0, 0>();
  jac.template get<1, 0>() = rotation_matrix_.template get<1, 0>();
  jac.template get<0, 1>() = rotation_matrix_.template get<0, 1>();
  jac.template get<1, 1>() = rotation_matrix_.template get<1, 1>();
  return jac;
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Rotation<2>::inv_jacobian(const std::array<T, 2>& /*xi*/) const {
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
      inv_jac{};
  inv_jac.template get<0, 0>() = rotation_matrix_.template get<0, 0>();
  inv_jac.template get<1, 0>() = rotation_matrix_.template get<0, 1>();
  inv_jac.template get<0, 1>() = rotation_matrix_.template get<1, 0>();
  inv_jac.template get<1, 1>() = rotation_matrix_.template get<1, 1>();
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
  rotation_matrix_.template get<0, 0>() =
      cos_gamma * cos_beta * cos_alpha - sin_gamma * sin_alpha;
  rotation_matrix_.template get<0, 1>() =
      -sin_gamma * cos_beta * cos_alpha - cos_gamma * sin_alpha;
  rotation_matrix_.template get<0, 2>() = sin_beta * cos_alpha;
  rotation_matrix_.template get<1, 0>() =
      cos_gamma * cos_beta * sin_alpha + sin_gamma * cos_alpha;
  rotation_matrix_.template get<1, 1>() =
      -sin_gamma * cos_beta * sin_alpha + cos_gamma * cos_alpha;
  rotation_matrix_.template get<1, 2>() = sin_beta * sin_alpha;
  rotation_matrix_.template get<2, 0>() = -cos_gamma * sin_beta;
  rotation_matrix_.template get<2, 1>() = sin_gamma * sin_beta;
  rotation_matrix_.template get<2, 2>() = cos_beta;
}

template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 3> Rotation<3>::
operator()(const std::array<T, 3>& xi) const {
  return {{xi[0] * rotation_matrix_.template get<0, 0>() +
               xi[1] * rotation_matrix_.template get<0, 1>() +
               xi[2] * rotation_matrix_.template get<0, 2>(),
           xi[0] * rotation_matrix_.template get<1, 0>() +
               xi[1] * rotation_matrix_.template get<1, 1>() +
               xi[2] * rotation_matrix_.template get<1, 2>(),
           xi[0] * rotation_matrix_.template get<2, 0>() +
               xi[1] * rotation_matrix_.template get<2, 1>() +
               xi[2] * rotation_matrix_.template get<2, 2>()}};
}

template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 3>
Rotation<3>::inverse(const std::array<T, 3>& x) const {
  // Inverse rotation matrix is the same as the transpose.
  return {{x[0] * rotation_matrix_.template get<0, 0>() +
               x[1] * rotation_matrix_.template get<1, 0>() +
               x[2] * rotation_matrix_.template get<2, 0>(),
           x[0] * rotation_matrix_.template get<0, 1>() +
               x[1] * rotation_matrix_.template get<1, 1>() +
               x[2] * rotation_matrix_.template get<2, 1>(),
           x[0] * rotation_matrix_.template get<0, 2>() +
               x[1] * rotation_matrix_.template get<1, 2>() +
               x[2] * rotation_matrix_.template get<2, 2>()}};
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Rotation<3>::jacobian(const std::array<T, 3>& /*xi*/) const {
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
      jac{};
  jac.template get<0, 0>() = rotation_matrix_.template get<0, 0>();
  jac.template get<1, 0>() = rotation_matrix_.template get<1, 0>();
  jac.template get<0, 1>() = rotation_matrix_.template get<0, 1>();
  jac.template get<1, 1>() = rotation_matrix_.template get<1, 1>();
  jac.template get<2, 0>() = rotation_matrix_.template get<2, 0>();
  jac.template get<2, 1>() = rotation_matrix_.template get<2, 1>();
  jac.template get<0, 2>() = rotation_matrix_.template get<0, 2>();
  jac.template get<1, 2>() = rotation_matrix_.template get<1, 2>();
  jac.template get<2, 2>() = rotation_matrix_.template get<2, 2>();
  return jac;
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Rotation<3>::inv_jacobian(const std::array<T, 3>& /*xi*/) const {
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
      inv_jac{};
  inv_jac.template get<0, 0>() = rotation_matrix_.template get<0, 0>();
  inv_jac.template get<1, 0>() = rotation_matrix_.template get<0, 1>();
  inv_jac.template get<0, 1>() = rotation_matrix_.template get<1, 0>();
  inv_jac.template get<1, 1>() = rotation_matrix_.template get<1, 1>();
  inv_jac.template get<2, 0>() = rotation_matrix_.template get<0, 2>();
  inv_jac.template get<2, 1>() = rotation_matrix_.template get<1, 2>();
  inv_jac.template get<0, 2>() = rotation_matrix_.template get<2, 0>();
  inv_jac.template get<1, 2>() = rotation_matrix_.template get<2, 1>();
  inv_jac.template get<2, 2>() = rotation_matrix_.template get<2, 2>();
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
    const std::array<std::reference_wrapper<const double>, 2>& /*xi*/) const;
template std::array<double, 2> Rotation<2>::operator()(
    const std::array<double, 2>& /*xi*/) const;
template std::array<DataVector, 2> Rotation<2>::operator()(
    const std::array<std::reference_wrapper<const DataVector>, 2>& /*xi*/)
    const;
template std::array<DataVector, 2> Rotation<2>::operator()(
    const std::array<DataVector, 2>& /*xi*/) const;

template std::array<double, 2> Rotation<2>::inverse(
    const std::array<std::reference_wrapper<const double>, 2>& /*xi*/) const;
template std::array<double, 2> Rotation<2>::inverse(
    const std::array<double, 2>& /*xi*/) const;
template std::array<DataVector, 2> Rotation<2>::inverse(
    const std::array<std::reference_wrapper<const DataVector>, 2>& /*xi*/)
    const;
template std::array<DataVector, 2> Rotation<2>::inverse(
    const std::array<DataVector, 2>& /*xi*/) const;

template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Rotation<2>::jacobian(
    const std::array<std::reference_wrapper<const double>, 2>& /*xi*/) const;
template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Rotation<2>::jacobian(const std::array<double, 2>& /*xi*/) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Rotation<2>::jacobian(const std::array<std::reference_wrapper<const DataVector>,
                                       2>& /*xi*/) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Rotation<2>::jacobian(const std::array<DataVector, 2>& /*xi*/) const;

template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Rotation<2>::inv_jacobian(
    const std::array<std::reference_wrapper<const double>, 2>& /*xi*/) const;
template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Rotation<2>::inv_jacobian(const std::array<double, 2>& /*xi*/) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Rotation<2>::inv_jacobian(
    const std::array<std::reference_wrapper<const DataVector>, 2>& /*xi*/)
    const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Rotation<2>::inv_jacobian(const std::array<DataVector, 2>& /*xi*/) const;

template std::array<double, 3> Rotation<3>::operator()(
    const std::array<std::reference_wrapper<const double>, 3>& xi) const;
template std::array<double, 3> Rotation<3>::operator()(
    const std::array<double, 3>& xi) const;
template std::array<DataVector, 3> Rotation<3>::operator()(
    const std::array<std::reference_wrapper<const DataVector>, 3>& xi) const;
template std::array<DataVector, 3> Rotation<3>::operator()(
    const std::array<DataVector, 3>& xi) const;

template std::array<double, 3> Rotation<3>::inverse(
    const std::array<std::reference_wrapper<const double>, 3>& xi) const;
template std::array<double, 3> Rotation<3>::inverse(
    const std::array<double, 3>& xi) const;
template std::array<DataVector, 3> Rotation<3>::inverse(
    const std::array<std::reference_wrapper<const DataVector>, 3>& xi) const;
template std::array<DataVector, 3> Rotation<3>::inverse(
    const std::array<DataVector, 3>& xi) const;

template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Rotation<3>::jacobian(
    const std::array<std::reference_wrapper<const double>, 3>& /*xi*/) const;
template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Rotation<3>::jacobian(const std::array<double, 3>& /*xi*/) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Rotation<3>::jacobian(const std::array<std::reference_wrapper<const DataVector>,
                                       3>& /*xi*/) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Rotation<3>::jacobian(const std::array<DataVector, 3>& /*xi*/) const;

template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Rotation<3>::inv_jacobian(
    const std::array<std::reference_wrapper<const double>, 3>& /*xi*/) const;
template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Rotation<3>::inv_jacobian(const std::array<double, 3>& /*xi*/) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Rotation<3>::inv_jacobian(
    const std::array<std::reference_wrapper<const DataVector>, 3>& /*xi*/)
    const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Rotation<3>::inv_jacobian(const std::array<DataVector, 3>& /*xi*/) const;

}  // namespace CoordinateMaps
