// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Wedge3D.hpp"

#include "DataStructures/MakeWithValue.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace CoordinateMaps {

Wedge3D::Wedge3D(const double radius_of_other_surface,
                 const double radius_of_spherical_surface,
                 const Direction<3> direction_of_wedge,
                 const double sphericity_of_other_surface,
                 const bool with_equiangular_map) noexcept
    : radius_of_other_surface_(radius_of_other_surface),
      radius_of_spherical_surface_(radius_of_spherical_surface),
      direction_of_wedge_(direction_of_wedge),
      sphericity_of_other_surface_(sphericity_of_other_surface),
      with_equiangular_map_(with_equiangular_map) {
  ASSERT(radius_of_other_surface > 0,
         "The radius of the other surface must be greater than zero.");
  ASSERT(radius_of_spherical_surface > 0,
         "The radius of the spherical surface must be greater than zero.");
  ASSERT(sphericity_of_other_surface >= 0 and sphericity_of_other_surface <= 1,
         "Sphericity of other surface must be between 0 and 1");
  ASSERT(radius_of_other_surface < radius_of_spherical_surface or
             (1 + sphericity_of_other_surface * (sqrt(3.) - 1.)) *
                     radius_of_other_surface >
                 sqrt(3.) * radius_of_spherical_surface,
         "For the value of the given radii and sphericity, the spherical "
         "surface intersects with the other surface");
}

template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 3> Wedge3D::
operator()(const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = std::decay_t<tt::remove_reference_wrapper_t<T>>;

  const ReturnType cap_xi =
      with_equiangular_map_
          ? tan(M_PI_4 * static_cast<ReturnType>(source_coords[0]))
          : static_cast<ReturnType>(source_coords[0]);
  const ReturnType cap_eta =
      with_equiangular_map_
          ? tan(M_PI_4 * static_cast<ReturnType>(source_coords[1]))
          : static_cast<ReturnType>(source_coords[1]);
  const ReturnType& zeta = source_coords[2];

  const ReturnType first_blending_factor =
      0.5 * (1.0 - sphericity_of_other_surface_) * radius_of_other_surface_ /
      sqrt(3.0) * (1.0 - zeta);
  const ReturnType second_blending_factor_over_rho =
      (0.5 * sphericity_of_other_surface_ * radius_of_other_surface_ *
           (1.0 - zeta) +
       0.5 * radius_of_spherical_surface_ * (1.0 + zeta)) /
      sqrt(1.0 + square(cap_xi) + square(cap_eta));

  ReturnType physical_z =
      first_blending_factor + second_blending_factor_over_rho;
  ReturnType physical_x = physical_z * cap_xi;
  ReturnType physical_y = physical_z * cap_eta;

  // Assigns location of wedge based on passed "direction_of_wedge"
  // Wedges on z axis:
  if (direction_of_wedge_.axis() == Direction<3>::Axis::Zeta) {
    return (direction_of_wedge_.side() == Side::Upper)
               ? std::array<ReturnType, 3>{{std::move(physical_x),
                                            std::move(physical_y),
                                            std::move(physical_z)}}
               : std::array<ReturnType, 3>{{std::move(physical_x),
                                            std::move(-physical_y),
                                            std::move(-physical_z)}};
  }
  // Wedges on y axis:
  if (direction_of_wedge_.axis() == Direction<3>::Axis::Eta) {
    return (direction_of_wedge_.side() == Side::Upper)
               ? std::array<ReturnType, 3>{{std::move(physical_y),
                                            std::move(physical_z),
                                            std::move(physical_x)}}
               : std::array<ReturnType, 3>{{std::move(physical_y),
                                            std::move(-physical_z),
                                            std::move(-physical_x)}};
  }
  // Wedges on x axis:
  if (direction_of_wedge_.axis() == Direction<3>::Axis::Xi) {
    return (direction_of_wedge_.side() == Side::Upper)
               ? std::array<ReturnType, 3>{{std::move(physical_z),
                                            std::move(physical_x),
                                            std::move(physical_y)}}
               : std::array<ReturnType, 3>{{std::move(-physical_z),
                                            std::move(-physical_x),
                                            std::move(physical_y)}};
  }
  ERROR("Improper axis passed to function.");
}

template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 3> Wedge3D::inverse(
    const std::array<T, 3>& target_coords) const noexcept {
  using ReturnType = std::decay_t<tt::remove_reference_wrapper_t<T>>;

  ReturnType physical_x;
  ReturnType physical_y;
  ReturnType physical_z;

  // Assigns location of wedge based on passed "direction_of_wedge"
  // Wedges on z axis:
  switch (direction_of_wedge_.axis()) {
    case Direction<3>::Axis::Zeta:
      if (direction_of_wedge_.side() == Side::Upper) {
        physical_x = target_coords[0];
        physical_y = target_coords[1];
        physical_z = target_coords[2];
      } else {
        physical_x = target_coords[0];
        physical_y = -target_coords[1];
        physical_z = -target_coords[2];
      }
      break;
      // Wedges on y axis:
    case Direction<3>::Axis::Eta:
      if (direction_of_wedge_.side() == Side::Upper) {
        physical_x = target_coords[2];
        physical_y = target_coords[0];
        physical_z = target_coords[1];
      } else {
        physical_x = -target_coords[2];
        physical_y = target_coords[0];
        physical_z = -target_coords[1];
      }
      break;
      // Wedges on x axis:
    case Direction<3>::Axis::Xi:
      if (direction_of_wedge_.side() == Side::Upper) {
        physical_x = target_coords[1];
        physical_y = target_coords[2];
        physical_z = target_coords[0];
      } else {
        physical_x = -target_coords[1];
        physical_y = target_coords[2];
        physical_z = -target_coords[0];
      }
      break;
    default:
      ERROR("Switch failed... somehow?.");
  }

  ReturnType cap_xi = physical_x / physical_z;
  ReturnType cap_eta = physical_y / physical_z;
  ReturnType one_over_rho = 1.0 / sqrt(1.0 + square(cap_xi) + square(cap_eta));
  ReturnType common_factor_one =
      0.5 * radius_of_other_surface_ *
      ((1.0 - sphericity_of_other_surface_) / sqrt(3.0) +
       sphericity_of_other_surface_ * one_over_rho);
  ReturnType common_factor_two =
      0.5 * radius_of_spherical_surface_ * one_over_rho;
  ReturnType zeta = (physical_z - (common_factor_two + common_factor_one)) /
                    (common_factor_two - common_factor_one);
  ReturnType xi =
      with_equiangular_map_ ? atan(cap_xi) / M_PI_4 : std::move(cap_xi);
  ReturnType eta =
      with_equiangular_map_ ? atan(cap_eta) / M_PI_4 : std::move(cap_eta);
  return std::array<ReturnType, 3>{
      {std::move(xi), std::move(eta), std::move(zeta)}};
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Wedge3D::jacobian(const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = std::decay_t<tt::remove_reference_wrapper_t<T>>;

  const ReturnType& xi = source_coords[0];
  const ReturnType& eta = source_coords[1];
  const ReturnType& zeta = source_coords[2];

  const ReturnType& cap_xi = with_equiangular_map_ ? tan(M_PI_4 * xi) : xi;
  const ReturnType& cap_eta = with_equiangular_map_ ? tan(M_PI_4 * eta) : eta;

  const ReturnType cap_xi_deriv = with_equiangular_map_
                                      ? M_PI_4 * (1 + square(cap_xi))
                                      : make_with_value<ReturnType>(xi, 1.0);
  const ReturnType cap_eta_deriv = with_equiangular_map_
                                       ? M_PI_4 * (1 + square(cap_eta))
                                       : make_with_value<ReturnType>(eta, 1.0);

  const ReturnType one_over_rho =
      1.0 / sqrt(1.0 + square(cap_xi) + square(cap_eta));
  const ReturnType first_blending_factor =
      0.5 * (1.0 - sphericity_of_other_surface_) * radius_of_other_surface_ /
      sqrt(3.0) * (1.0 - zeta);
  const double first_blending_rate = -0.5 *
                                     (1.0 - sphericity_of_other_surface_) *
                                     radius_of_other_surface_ / sqrt(3.0);
  const ReturnType second_blending_factor_over_rho_cubed =
      (0.5 * sphericity_of_other_surface_ * radius_of_other_surface_ *
           (1.0 - zeta) +
       0.5 * radius_of_spherical_surface_ * (1.0 + zeta)) *
      pow<3>(one_over_rho);
  const ReturnType second_blending_rate_over_rho =
      0.5 *
      (-sphericity_of_other_surface_ * radius_of_other_surface_ +
       radius_of_spherical_surface_) *
      one_over_rho;

  Tensor<ReturnType, tmpl::integral_list<std::int32_t, 2, 1>,
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

  dz_dxi = -second_blending_factor_over_rho_cubed * cap_xi * cap_xi_deriv;
  dz_deta = -second_blending_factor_over_rho_cubed * cap_eta * cap_eta_deriv;
  dz_dzeta = first_blending_rate + second_blending_rate_over_rho;
  dx_dxi = (first_blending_factor +
            second_blending_factor_over_rho_cubed * (1.0 + square(cap_eta))) *
           cap_xi_deriv;
  dx_deta = dz_deta * cap_xi;
  dx_dzeta = dz_dzeta * cap_xi;
  dy_dxi = dz_dxi * cap_eta;
  dy_deta = (first_blending_factor +
             second_blending_factor_over_rho_cubed * (1.0 + square(cap_xi))) *
            cap_eta_deriv;
  dy_dzeta = dz_dzeta * cap_eta;

  // Implement Rotation:
  if (direction_of_wedge_ == Direction<3>::lower_zeta()) {
    dz_dxi *= -1;
    dz_deta *= -1;
    dz_dzeta *= -1;

    dy_dxi *= -1;
    dy_deta *= -1;
    dy_dzeta *= -1;
  }

  if (direction_of_wedge_.axis() == Direction<3>::Axis::Eta) {
    std::swap(dx_dxi, dy_dxi);
    std::swap(dx_deta, dy_deta);
    std::swap(dx_dzeta, dy_dzeta);
    std::swap(dy_dxi, dz_dxi);
    std::swap(dy_deta, dz_deta);
    std::swap(dy_dzeta, dz_dzeta);

    if (direction_of_wedge_.side() == Side::Lower) {
      dy_dxi *= -1;
      dy_deta *= -1;
      dy_dzeta *= -1;

      dz_dxi *= -1;
      dz_deta *= -1;
      dz_dzeta *= -1;
    }
  }

  if (direction_of_wedge_.axis() == Direction<3>::Axis::Xi) {
    std::swap(dx_dxi, dz_dxi);
    std::swap(dx_deta, dz_deta);
    std::swap(dx_dzeta, dz_dzeta);
    std::swap(dz_dxi, dy_dxi);
    std::swap(dz_deta, dy_deta);
    std::swap(dz_dzeta, dy_dzeta);

    if (direction_of_wedge_.side() == Side::Lower) {
      dx_dxi *= -1;
      dx_deta *= -1;
      dx_dzeta *= -1;

      dy_dxi *= -1;
      dy_deta *= -1;
      dy_dzeta *= -1;
    }
  }

  return jacobian_matrix;
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Wedge3D::inv_jacobian(const std::array<T, 3>& source_coords) const noexcept {
  return determinant_and_inverse(jacobian(source_coords)).second;
}

void Wedge3D::pup(PUP::er& p) noexcept {
  p | radius_of_other_surface_;
  p | radius_of_spherical_surface_;
  p | direction_of_wedge_;
  p | sphericity_of_other_surface_;
  p | with_equiangular_map_;
}

bool operator==(const Wedge3D& lhs, const Wedge3D& rhs) noexcept {
  return lhs.radius_of_other_surface_ == rhs.radius_of_other_surface_ and
         lhs.radius_of_spherical_surface_ ==
             rhs.radius_of_spherical_surface_ and
         lhs.direction_of_wedge_ == rhs.direction_of_wedge_ and
         lhs.sphericity_of_other_surface_ ==
             rhs.sphericity_of_other_surface_ and
         lhs.with_equiangular_map_ == rhs.with_equiangular_map_;
}

bool operator!=(const Wedge3D& lhs, const Wedge3D& rhs) noexcept {
  return not(lhs == rhs);
}

// Explicit instantiations
template std::array<double, 3> Wedge3D::operator()(
    const std::array<std::reference_wrapper<const double>, 3>& source_coords)
    const noexcept;
template std::array<double, 3> Wedge3D::operator()(
    const std::array<double, 3>& source_coords) const noexcept;
template std::array<DataVector, 3> Wedge3D::operator()(
    const std::array<std::reference_wrapper<const DataVector>, 3>&
        source_coords) const noexcept;
template std::array<DataVector, 3> Wedge3D::operator()(
    const std::array<DataVector, 3>& source_coords) const noexcept;

template std::array<double, 3> Wedge3D::inverse(
    const std::array<std::reference_wrapper<const double>, 3>& target_coords)
    const noexcept;
template std::array<double, 3> Wedge3D::inverse(
    const std::array<double, 3>& target_coords) const noexcept;
template std::array<DataVector, 3> Wedge3D::inverse(
    const std::array<std::reference_wrapper<const DataVector>, 3>&
        target_coords) const noexcept;
template std::array<DataVector, 3> Wedge3D::inverse(
    const std::array<DataVector, 3>& target_coords) const noexcept;

template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Wedge3D::jacobian(const std::array<std::reference_wrapper<const double>, 3>&
                      source_coords) const noexcept;
template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Wedge3D::jacobian(const std::array<double, 3>& source_coords) const noexcept;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Wedge3D::jacobian(const std::array<std::reference_wrapper<const DataVector>, 3>&
                      source_coords) const noexcept;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Wedge3D::jacobian(const std::array<DataVector, 3>& source_coords) const
    noexcept;

template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Wedge3D::inv_jacobian(const std::array<std::reference_wrapper<const double>, 3>&
                          source_coords) const noexcept;
template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Wedge3D::inv_jacobian(const std::array<double, 3>& source_coords) const
    noexcept;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Wedge3D::inv_jacobian(const std::array<std::reference_wrapper<const DataVector>,
                                       3>& source_coords) const noexcept;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Wedge3D::inv_jacobian(const std::array<DataVector, 3>& source_coords) const
    noexcept;
}  // namespace CoordinateMaps
