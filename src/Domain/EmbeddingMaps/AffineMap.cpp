// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/EmbeddingMaps/AffineMap.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

namespace EmbeddingMaps {

AffineMap::AffineMap(const double A, const double B, const double a,
                     const double b)
    : A_(A),
      B_(B),
      a_(a),
      b_(b),
      length_of_domain_(B - A),
      length_of_range_(b - a),
      jacobian_(length_of_range_ / length_of_domain_),
      inverse_jacobian_(length_of_domain_ / length_of_range_) {}

template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 1> AffineMap::
operator()(const std::array<T, 1>& xi) const {
  return {{(length_of_range_ * xi[0] + a_ * B_ - b_ * A_) / length_of_domain_}};
}

template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 1>
AffineMap::inverse(const std::array<T, 1>& x) const {
  return {{(length_of_domain_ * x[0] - a_ * B_ + b_ * A_) / length_of_range_}};
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>
AffineMap::jacobian(const std::array<T, 1>& /*xi*/) const {
  return Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
                tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>{
      jacobian_};
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>
AffineMap::inv_jacobian(const std::array<T, 1>& /*xi*/) const {
  return Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
                tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>{
      inverse_jacobian_};
}

void AffineMap::pup(PUP::er& p) {
  p | A_;
  p | B_;
  p | a_;
  p | b_;
  p | length_of_domain_;
  p | length_of_range_;
  p | jacobian_;
  p | inverse_jacobian_;
}

bool operator==(const EmbeddingMaps::AffineMap& lhs,
                const EmbeddingMaps::AffineMap& rhs) noexcept {
  return lhs.A_ == rhs.A_ and lhs.B_ == rhs.B_ and lhs.a_ == rhs.a_ and
         lhs.b_ == rhs.b_ and lhs.length_of_domain_ == rhs.length_of_domain_ and
         lhs.length_of_range_ == rhs.length_of_range_ and
         lhs.jacobian_ == rhs.jacobian_ and
         lhs.inverse_jacobian_ == rhs.inverse_jacobian_;
}

// Explicit instantiations
template std::array<double, 1> AffineMap::operator()(
    const std::array<std::reference_wrapper<const double>, 1>& /*xi*/) const;
template std::array<double, 1> AffineMap::operator()(
    const std::array<double, 1>& /*xi*/) const;
template std::array<DataVector, 1> AffineMap::operator()(
    const std::array<std::reference_wrapper<const DataVector>, 1>& /*xi*/)
    const;
template std::array<DataVector, 1> AffineMap::operator()(
    const std::array<DataVector, 1>& /*xi*/) const;

template std::array<double, 1> AffineMap::inverse(
    const std::array<std::reference_wrapper<const double>, 1>& /*xi*/) const;
template std::array<double, 1> AffineMap::inverse(
    const std::array<double, 1>& /*xi*/) const;
template std::array<DataVector, 1> AffineMap::inverse(
    const std::array<std::reference_wrapper<const DataVector>, 1>& /*xi*/)
    const;
template std::array<DataVector, 1> AffineMap::inverse(
    const std::array<DataVector, 1>& /*xi*/) const;

template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>
AffineMap::jacobian(
    const std::array<std::reference_wrapper<const double>, 1>& /*xi*/) const;
template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>
AffineMap::jacobian(const std::array<double, 1>& /*xi*/) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>
AffineMap::jacobian(const std::array<std::reference_wrapper<const DataVector>,
                                     1>& /*xi*/) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>
AffineMap::jacobian(const std::array<DataVector, 1>& /*xi*/) const;

template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>
AffineMap::inv_jacobian(
    const std::array<std::reference_wrapper<const double>, 1>& /*xi*/) const;
template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>
AffineMap::inv_jacobian(const std::array<double, 1>& /*xi*/) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>
AffineMap::inv_jacobian(
    const std::array<std::reference_wrapper<const DataVector>, 1>& /*xi*/)
    const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>
AffineMap::inv_jacobian(const std::array<DataVector, 1>& /*xi*/) const;
}  // namespace EmbeddingMaps
