// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Wedge2D.hpp"

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace CoordinateMaps {

Wedge2D::Wedge2D(double radius_of_other, double radius_of_circle,
                 Direction<2> direction_of_wedge,
                 bool with_equiangular_map) noexcept
    : radius_of_other_(radius_of_other),
      radius_of_circle_(radius_of_circle),
      direction_of_wedge_(direction_of_wedge),
      with_equiangular_map_(with_equiangular_map) {
  ASSERT(radius_of_other > 0.0, "This radius must be greater than zero.");
  ASSERT(radius_of_circle > 0.0, "This radius must be greater than zero.");
  ASSERT(radius_of_circle > radius_of_other or
             radius_of_other > sqrt(2.0) * radius_of_circle,
         "The faces of the wedge must not intersect.");
}

template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 2> Wedge2D::
operator()(const std::array<T, 2>& source_coords) const noexcept {
  using ReturnType = std::decay_t<tt::remove_reference_wrapper_t<T>>;

  const auto physical_coordinates = [this](const ReturnType& xi,
                                           const ReturnType& cap_eta) {
    const auto physical_x =
        0.5 * radius_of_other_ / sqrt(2.0) * (1.0 - xi) +
        0.5 * radius_of_circle_ / sqrt(1.0 + square(cap_eta)) * (1.0 + xi);
    const auto physical_y = cap_eta * physical_x;

    // Wedges on x axis:
    if (direction_of_wedge_.axis() == Direction<2>::Axis::Xi) {
      return (direction_of_wedge_.side() == Side::Upper)
                 ? std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>,
                              2>{{physical_x, physical_y}}
                 : std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>,
                              2>{{-physical_x, -physical_y}};
    }
    // Wedges on y axis:
    return (direction_of_wedge_.side() == Side::Upper)
               ? std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>,
                            2>{{-physical_y, physical_x}}
               : std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 2>{
                     {physical_y, -physical_x}};
  };

  if (with_equiangular_map_) {
    return physical_coordinates(
        dereference_wrapper(source_coords[0]),
        tan(M_PI_4 * dereference_wrapper(source_coords[1])));
  }
  return physical_coordinates(dereference_wrapper(source_coords[0]),
                              dereference_wrapper(source_coords[1]));
}

template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 2> Wedge2D::inverse(
    const std::array<T, 2>& target_coords) const noexcept {
  using ReturnType = std::decay_t<tt::remove_reference_wrapper_t<T>>;

  ReturnType physical_x;
  ReturnType physical_y;

  // Assigns location of wedge based on passed "direction_of_wedge"
  // Wedges on x axis:
  switch (direction_of_wedge_.axis()) {
    case Direction<2>::Axis::Xi:
      if (direction_of_wedge_.side() == Side::Upper) {
        physical_x = target_coords[0];
        physical_y = target_coords[1];
      } else {
        physical_x = -target_coords[0];
        physical_y = -target_coords[1];
      }
      break;
      // Wedges on y axis:
    case Direction<2>::Axis::Eta:
      if (direction_of_wedge_.side() == Side::Upper) {
        physical_x = target_coords[1];
        physical_y = -target_coords[0];
      } else {
        physical_x = -target_coords[1];
        physical_y = target_coords[0];
      }
      break;
    default:
      ERROR("Switch failed. `direction_of_wedge` must be a 2D axis direction.");
  }

  const ReturnType cap_eta = physical_y / physical_x;
  const ReturnType one_over_rho = 1.0 / sqrt(1.0 + square(cap_eta));
  const double common_factor_one = 0.5 * radius_of_other_ / sqrt(2.0);
  const ReturnType common_factor_two = 0.5 * radius_of_circle_ * one_over_rho;
  ReturnType xi = (physical_x - (common_factor_two + common_factor_one)) /
                  (common_factor_two - common_factor_one);
  ReturnType eta =
      with_equiangular_map_ ? atan(cap_eta) / M_PI_4 : std::move(cap_eta);
  return std::array<ReturnType, 2>{{std::move(xi), std::move(eta)}};
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Wedge2D::jacobian(const std::array<T, 2>& source_coords) const noexcept {
  using ReturnType = std::decay_t<tt::remove_reference_wrapper_t<T>>;
  const auto jacobian_lambda = [this](const ReturnType& xi,
                                      const ReturnType& cap_eta,
                                      const auto& cap_eta_deriv) {
    const ReturnType one_over_rho = 1.0 / sqrt(1.0 + square(cap_eta));

    Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
           tmpl::integral_list<std::int32_t, 2, 1>,
           index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                      SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
        jacobian_matrix{};

    auto &dxdxi = get<0, 0>(jacobian_matrix),
         &dydeta = get<1, 1>(jacobian_matrix),
         &dxdeta = get<0, 1>(jacobian_matrix),
         &dydxi = get<1, 0>(jacobian_matrix);

    dxdxi = -0.5 * radius_of_other_ / sqrt(2.0) +
            0.5 * radius_of_circle_ * one_over_rho;
    dxdeta = -0.5 * radius_of_circle_ * (1.0 + xi) * cap_eta * cap_eta_deriv *
             pow<3>(one_over_rho);
    dydxi = cap_eta * dxdxi;
    dydeta = cap_eta_deriv *
             (0.5 * radius_of_other_ / sqrt(2.0) * (1.0 - xi) +
              0.5 * radius_of_circle_ * (1.0 + xi) * pow<3>(one_over_rho));

    if (direction_of_wedge_.axis() == Direction<2>::Axis::Eta) {
      std::swap(dxdxi, dydxi);
      std::swap(dxdeta, dydeta);
      dxdxi *= -1.0;
      dxdeta *= -1.0;
    }
    if (direction_of_wedge_.side() == Side::Lower) {
      dxdxi *= -1.0;
      dxdeta *= -1.0;
      dydxi *= -1.0;
      dydeta *= -1.0;
    }
    return jacobian_matrix;
  };

  if (with_equiangular_map_) {
    return jacobian_lambda(
        dereference_wrapper(source_coords[0]),
        tan(M_PI_4 * dereference_wrapper(source_coords[1])),
        ReturnType{M_PI_4 *
                   (1.0 + square(tan(M_PI_4 *
                                     dereference_wrapper(source_coords[1]))))});
  }
  return jacobian_lambda(dereference_wrapper(source_coords[0]),
                         dereference_wrapper(source_coords[1]), 1.0);
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Wedge2D::inv_jacobian(const std::array<T, 2>& source_coords) const noexcept {
  const auto jac = jacobian(source_coords);
  return determinant_and_inverse(jac).second;
}

void Wedge2D::pup(PUP::er& p) {
  p | radius_of_other_;
  p | radius_of_circle_;
  p | direction_of_wedge_;
  p | with_equiangular_map_;
}

bool operator==(const Wedge2D& lhs, const Wedge2D& rhs) noexcept {
  return lhs.radius_of_other_ == rhs.radius_of_other_ and
         lhs.radius_of_circle_ == rhs.radius_of_circle_ and
         lhs.direction_of_wedge_ == rhs.direction_of_wedge_ and
         lhs.with_equiangular_map_ == rhs.with_equiangular_map_;
}

bool operator!=(const Wedge2D& lhs, const Wedge2D& rhs) noexcept {
  return not(lhs == rhs);
}

// Explicit instantiations
/// \cond
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                             \
  template std::array<DTYPE(data), 2> Wedge2D::operator()(               \
      const std::array<std::reference_wrapper<const DTYPE(data)>, 2>&    \
          source_coords) const noexcept;                                 \
  template std::array<DTYPE(data), 2> Wedge2D::operator()(               \
      const std::array<DTYPE(data), 2>& source_coords) const noexcept;   \
  template std::array<DTYPE(data), 2> Wedge2D::inverse(                  \
      const std::array<std::reference_wrapper<const DTYPE(data)>, 2>&    \
          target_coords) const noexcept;                                 \
  template std::array<DTYPE(data), 2> Wedge2D::inverse(                  \
      const std::array<DTYPE(data), 2>& target_coords) const noexcept;   \
  template Tensor<DTYPE(data), tmpl::integral_list<std::int32_t, 2, 1>,  \
                  index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,  \
                             SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>> \
  Wedge2D::jacobian(                                                     \
      const std::array<std::reference_wrapper<const DTYPE(data)>, 2>&    \
          source_coords) const noexcept;                                 \
  template Tensor<DTYPE(data), tmpl::integral_list<std::int32_t, 2, 1>,  \
                  index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,  \
                             SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>> \
  Wedge2D::jacobian(const std::array<DTYPE(data), 2>& source_coords)     \
      const noexcept;                                                    \
  template Tensor<DTYPE(data), tmpl::integral_list<std::int32_t, 2, 1>,  \
                  index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,  \
                             SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>> \
  Wedge2D::inv_jacobian(                                                 \
      const std::array<std::reference_wrapper<const DTYPE(data)>, 2>&    \
          source_coords) const noexcept;                                 \
  template Tensor<DTYPE(data), tmpl::integral_list<std::int32_t, 2, 1>,  \
                  index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,  \
                             SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>> \
  Wedge2D::inv_jacobian(const std::array<DTYPE(data), 2>& source_coords) \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
/// \endcond
}  // namespace CoordinateMaps
