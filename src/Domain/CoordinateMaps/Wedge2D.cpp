// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Wedge2D.hpp"

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"

namespace CoordinateMaps {

Wedge2D::Wedge2D(double inner_radius, double outer_radius,
                 Direction<2> positioning_of_wedge)
    : inner_radius_(inner_radius),
      outer_radius_(outer_radius),
      positioning_of_wedge_(positioning_of_wedge) {
  ASSERT(inner_radius > 0, "The inner radius must be greater than zero.");
  ASSERT(outer_radius > inner_radius,
         "The outer radius must be larger than the inner radius.");
  ASSERT(positioning_of_wedge_.axis() == Direction<2>::Axis::Xi or
             positioning_of_wedge_.axis() == Direction<2>::Axis::Eta,
         "The wedges must be located in the x or y directions.");
}

template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 2> Wedge2D::
operator()(const std::array<T, 2>& x) const noexcept {
  const auto theta = M_PI_4 * x[1];
  const auto physical_x = 0.5 * (1 - x[0]) * inner_radius_ / sqrt(2.0) +
                          0.5 * (1 + x[0]) * outer_radius_ * cos(theta);
  const auto physical_y = 0.5 * (1 - x[0]) * inner_radius_ * x[1] / sqrt(2.0) +
                          0.5 * (1 + x[0]) * outer_radius_ * sin(theta);

  // Wedges on x axis:
  if (positioning_of_wedge_.axis() == Direction<2>::Axis::Xi) {
    return (positioning_of_wedge_.side() == Side::Upper)
               ? std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>,
                            2>{{physical_x, physical_y}}
               : std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 2>{
                     {-physical_x, -physical_y}};
  }
  // Wedges on y axis:
  else {
    return (positioning_of_wedge_.side() == Side::Upper)
               ? std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>,
                            2>{{-physical_y, physical_x}}
               : std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 2>{
                     {physical_y, -physical_x}};
  }
}

template <typename T>
[[noreturn]] std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 2>
Wedge2D::inverse(const std::array<T, 2>& /*x*/) const noexcept {
  ERROR("Inverse map is unimplemented for Wedge2D"); //LCOV_EXCL_LINE
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Wedge2D::jacobian(const std::array<T, 2>& xi) const noexcept {
  const auto theta = M_PI_4 * xi[1];

  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
      jacobian_matrix{};

  auto &dxdxi = jacobian_matrix.template get<0, 0>(),
       &dydeta = jacobian_matrix.template get<1, 1>(),
       &dxdeta = jacobian_matrix.template get<0, 1>(),
       &dydxi = jacobian_matrix.template get<1, 0>();

  dxdxi = -0.5 * inner_radius_ / sqrt(2.0) + 0.5 * outer_radius_ * cos(theta);
  dxdeta = -0.5 * (1 + xi[0]) * outer_radius_ * sin(theta) * M_PI_4;
  dydxi = -0.5 * inner_radius_ * xi[1] / sqrt(2.0) +
          0.5 * outer_radius_ * sin(theta);
  dydeta = 0.5 * (1 - xi[0]) * inner_radius_ / sqrt(2.0) +
           0.5 * (1 + xi[0]) * outer_radius_ * cos(theta) * M_PI_4;
  if (positioning_of_wedge_.axis() == Direction<2>::Axis::Eta) {
    std::swap(dxdxi, dydxi);
    std::swap(dxdeta, dydeta);
    dxdxi *= -1;
    dxdeta *= -1;
  }
  if (positioning_of_wedge_.side() == Side::Lower) {
    dxdxi *= -1;
    dxdeta *= -1;
    dydxi *= -1;
    dydeta *= -1;
  }
  return jacobian_matrix;
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Wedge2D::inv_jacobian(const std::array<T, 2>& xi) const noexcept {
  const auto jac = jacobian(xi);
  return determinant_and_inverse(jac).second;
}

void Wedge2D::pup(PUP::er& p) {
  p | inner_radius_;
  p | outer_radius_;
  p | positioning_of_wedge_;
}

bool operator==(const Wedge2D& lhs, const Wedge2D& rhs) noexcept {
  return lhs.inner_radius_ == rhs.inner_radius_ and
         lhs.outer_radius_ == rhs.outer_radius_ and
         lhs.positioning_of_wedge_ == rhs.positioning_of_wedge_;
}

bool operator!=(const Wedge2D& lhs, const Wedge2D& rhs) noexcept {
  return not(lhs == rhs);
}

// Explicit instantiations
template std::array<double, 2> Wedge2D::operator()(
    const std::array<std::reference_wrapper<const double>, 2>& /*xi*/) const
    noexcept;
template std::array<double, 2> Wedge2D::operator()(
    const std::array<double, 2>& /*xi*/) const noexcept;
template std::array<DataVector, 2> Wedge2D::operator()(
    const std::array<std::reference_wrapper<const DataVector>, 2>& /*xi*/) const
    noexcept;
template std::array<DataVector, 2> Wedge2D::operator()(
    const std::array<DataVector, 2>& /*xi*/) const noexcept;

template std::array<double, 2> Wedge2D::inverse(
    const std::array<std::reference_wrapper<const double>, 2>& /*xi*/) const
    noexcept;
template std::array<double, 2> Wedge2D::inverse(
    const std::array<double, 2>& /*xi*/) const noexcept;
template std::array<DataVector, 2> Wedge2D::inverse(
    const std::array<std::reference_wrapper<const DataVector>, 2>& /*xi*/) const
    noexcept;
template std::array<DataVector, 2> Wedge2D::inverse(
    const std::array<DataVector, 2>& /*xi*/) const noexcept;

template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Wedge2D::jacobian(
    const std::array<std::reference_wrapper<const double>, 2>& /*xi*/) const
    noexcept;
template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Wedge2D::jacobian(const std::array<double, 2>& /*xi*/) const noexcept;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Wedge2D::jacobian(
    const std::array<std::reference_wrapper<const DataVector>, 2>& /*xi*/) const
    noexcept;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Wedge2D::jacobian(const std::array<DataVector, 2>& /*xi*/) const noexcept;

template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Wedge2D::inv_jacobian(
    const std::array<std::reference_wrapper<const double>, 2>& /*xi*/) const
    noexcept;
template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Wedge2D::inv_jacobian(const std::array<double, 2>& /*xi*/) const noexcept;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Wedge2D::inv_jacobian(
    const std::array<std::reference_wrapper<const DataVector>, 2>& /*xi*/) const
    noexcept;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Wedge2D::inv_jacobian(const std::array<DataVector, 2>& /*xi*/) const noexcept;
}  // namespace CoordinateMaps
