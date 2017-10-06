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

Trilinear::Trilinear(const std::array<std::array<double, 3>, 8> vertices)
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

  jacobian_matrix.template get<0, 0>() = dXdxi[0];
  jacobian_matrix.template get<0, 1>() = dXdeta[0];
  jacobian_matrix.template get<0, 2>() = dXdzeta[0];
  jacobian_matrix.template get<1, 0>() = dXdxi[1];
  jacobian_matrix.template get<1, 1>() = dXdeta[1];
  jacobian_matrix.template get<1, 2>() = dXdzeta[1];
  jacobian_matrix.template get<2, 0>() = dXdxi[2];
  jacobian_matrix.template get<2, 1>() = dXdeta[2];
  jacobian_matrix.template get<2, 2>() = dXdzeta[2];
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
