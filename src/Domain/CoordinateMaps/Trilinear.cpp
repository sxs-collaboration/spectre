// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Trilinear.hpp"

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace CoordinateMaps {

template <typename T, size_t dim>
std::array<T, dim> product_of(const T& t,
                              const std::array<double, dim>& coordinate_point) {
  std::array<T, dim> result;
  for (size_t i = 0; i < dim; i++) {
    result[i] = t * coordinate_point[i];
  }
  return result;
}

Trilinear::Trilinear(std::array<std::array<double, 3>, 8> vertices) noexcept
    : vertices_(std::move(vertices)) {}

template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 3> Trilinear::
operator()(const std::array<T, 3>& x) const noexcept {
  return 0.125 * (product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                                 (1.0 - x[0]) * (1.0 - x[1]) * (1.0 - x[2])),
                             vertices_[0]) +
                  product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                                 (1.0 + x[0]) * (1.0 - x[1]) * (1.0 - x[2])),
                             vertices_[1]) +
                  product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                                 (1.0 - x[0]) * (1.0 + x[1]) * (1.0 - x[2])),
                             vertices_[2]) +
                  product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                                 (1.0 + x[0]) * (1.0 + x[1]) * (1.0 - x[2])),
                             vertices_[3]) +
                  product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                                 (1.0 - x[0]) * (1.0 - x[1]) * (1.0 + x[2])),
                             vertices_[4]) +
                  product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                                 (1.0 + x[0]) * (1.0 - x[1]) * (1.0 + x[2])),
                             vertices_[5]) +
                  product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                                 (1.0 - x[0]) * (1.0 + x[1]) * (1.0 + x[2])),
                             vertices_[6]) +
                  product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                                 (1.0 + x[0]) * (1.0 + x[1]) * (1.0 + x[2])),
                             vertices_[7]));
}

template <typename T>
[[noreturn]] std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 3>
Trilinear::inverse(const std::array<T, 3>& /*x*/) const noexcept {
  ERROR("Function not implemented.");
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Trilinear::jacobian(const std::array<T, 3>& x) const noexcept {
  const auto dXdxi =
      0.125 * (product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              -(1.0 - x[1]) * (1.0 - x[2])),
                          vertices_[0]) +
               product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              (1.0 - x[1]) * (1.0 - x[2])),
                          vertices_[1]) +
               product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              -(1.0 + x[1]) * (1.0 - x[2])),
                          vertices_[2]) +
               product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              (1.0 + x[1]) * (1.0 - x[2])),
                          vertices_[3]) +
               product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              -(1.0 - x[1]) * (1.0 + x[2])),
                          vertices_[4]) +
               product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              (1.0 - x[1]) * (1.0 + x[2])),
                          vertices_[5]) +
               product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              -(1.0 + x[1]) * (1.0 + x[2])),
                          vertices_[6]) +
               product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              (1.0 + x[1]) * (1.0 + x[2])),
                          vertices_[7]));

  const auto dXdeta =
      0.125 * (product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              -(1.0 - x[0]) * (1.0 - x[2])),
                          vertices_[0]) +
               product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              -(1.0 + x[0]) * (1.0 - x[2])),
                          vertices_[1]) +
               product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              (1.0 - x[0]) * (1.0 - x[2])),
                          vertices_[2]) +
               product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              (1.0 + x[0]) * (1.0 - x[2])),
                          vertices_[3]) +
               product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              -(1.0 - x[0]) * (1.0 + x[2])),
                          vertices_[4]) +
               product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              -(1.0 + x[0]) * (1.0 + x[2])),
                          vertices_[5]) +
               product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              (1.0 - x[0]) * (1.0 + x[2])),
                          vertices_[6]) +
               product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              (1.0 + x[0]) * (1.0 + x[2])),
                          vertices_[7]));

  const auto dXdzeta =
      0.125 * (product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              -(1.0 - x[0]) * (1.0 - x[1])),
                          vertices_[0]) +
               product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              -(1.0 + x[0]) * (1.0 - x[1])),
                          vertices_[1]) +
               product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              -(1.0 - x[0]) * (1.0 + x[1])),
                          vertices_[2]) +
               product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              -(1.0 + x[0]) * (1.0 + x[1])),
                          vertices_[3]) +
               product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              (1.0 - x[0]) * (1.0 - x[1])),
                          vertices_[4]) +
               product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              (1.0 + x[0]) * (1.0 - x[1])),
                          vertices_[5]) +
               product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              (1.0 - x[0]) * (1.0 + x[1])),
                          vertices_[6]) +
               product_of(std::decay_t<tt::remove_reference_wrapper_t<T>>(
                              (1.0 + x[0]) * (1.0 + x[1])),
                          vertices_[7]));

  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
      jacobian_matrix{};

  auto &dx_dxi = get<0, 0>(jacobian_matrix),
       &dx_deta = get<0, 1>(jacobian_matrix),
       &dx_dzeta = get<0, 2>(jacobian_matrix),
       &dy_dxi = get<1, 0>(jacobian_matrix),
       &dy_deta = get<1, 1>(jacobian_matrix),
       &dy_dzeta = get<1, 2>(jacobian_matrix),
       &dz_dxi = get<2, 0>(jacobian_matrix),
       &dz_deta = get<2, 1>(jacobian_matrix),
       &dz_dzeta = get<2, 2>(jacobian_matrix);

  dx_dxi = dXdxi[0];
  dx_deta = dXdeta[0];
  dx_dzeta = dXdzeta[0];
  dy_dxi = dXdxi[1];
  dy_deta = dXdeta[1];
  dy_dzeta = dXdzeta[1];
  dz_dxi = dXdxi[2];
  dz_deta = dXdeta[2];
  dz_dzeta = dXdzeta[2];
  return jacobian_matrix;
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Trilinear::inv_jacobian(const std::array<T, 3>& xi) const noexcept {
  const auto jac = jacobian(xi);
  return determinant_and_inverse(jac).second;
}

void Trilinear::pup(PUP::er& p) { p | vertices_; }

bool operator==(const Trilinear& lhs, const Trilinear& rhs) noexcept {
  return lhs.vertices_ == rhs.vertices_;
}

bool operator!=(const Trilinear& lhs, const Trilinear& rhs) noexcept {
  return not(lhs == rhs);
}

// Explicit instantiations
template std::array<double, 3> Trilinear::operator()(
    const std::array<std::reference_wrapper<const double>, 3>& /*xi*/) const
    noexcept;
template std::array<double, 3> Trilinear::operator()(
    const std::array<double, 3>& /*xi*/) const noexcept;
template std::array<DataVector, 3> Trilinear::operator()(
    const std::array<std::reference_wrapper<const DataVector>, 3>& /*xi*/) const
    noexcept;
template std::array<DataVector, 3> Trilinear::operator()(
    const std::array<DataVector, 3>& /*xi*/) const noexcept;

template std::array<double, 3> Trilinear::inverse(
    const std::array<std::reference_wrapper<const double>, 3>& /*xi*/) const
    noexcept;
template std::array<double, 3> Trilinear::inverse(
    const std::array<double, 3>& /*xi*/) const noexcept;
template std::array<DataVector, 3> Trilinear::inverse(
    const std::array<std::reference_wrapper<const DataVector>, 3>& /*xi*/) const
    noexcept;
template std::array<DataVector, 3> Trilinear::inverse(
    const std::array<DataVector, 3>& /*xi*/) const noexcept;

template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Trilinear::jacobian(
    const std::array<std::reference_wrapper<const double>, 3>& /*xi*/) const
    noexcept;
template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Trilinear::jacobian(const std::array<double, 3>& /*xi*/) const noexcept;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Trilinear::jacobian(
    const std::array<std::reference_wrapper<const DataVector>, 3>& /*xi*/) const
    noexcept;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Trilinear::jacobian(const std::array<DataVector, 3>& /*xi*/) const noexcept;

template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Trilinear::inv_jacobian(
    const std::array<std::reference_wrapper<const double>, 3>& /*xi*/) const
    noexcept;
template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Trilinear::inv_jacobian(const std::array<double, 3>& /*xi*/) const noexcept;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Trilinear::inv_jacobian(
    const std::array<std::reference_wrapper<const DataVector>, 3>& /*xi*/) const
    noexcept;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Trilinear::inv_jacobian(const std::array<DataVector, 3>& /*xi*/) const noexcept;
}  // namespace CoordinateMaps
