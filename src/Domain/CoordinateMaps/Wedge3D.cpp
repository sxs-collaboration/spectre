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
                 const double sphericity_of_other_surface) noexcept
    : radius_of_other_surface_(radius_of_other_surface),
      radius_of_spherical_surface_(radius_of_spherical_surface),
      direction_of_wedge_(direction_of_wedge),
      sphericity_of_other_surface_(sphericity_of_other_surface) {
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
operator()(const std::array<T, 3>& x) const noexcept {
  using ReturnType = std::decay_t<tt::remove_reference_wrapper_t<T>>;

  const ReturnType& xi = x[0];
  const ReturnType& eta = x[1];
  const ReturnType& zeta = x[2];
  ReturnType x_inner_cubical_face = xi * radius_of_other_surface_ / sqrt(3.0);
  ReturnType y_inner_cubical_face = eta * radius_of_other_surface_ / sqrt(3.0);
  // clang-tidy: use auto - This use would be OK, but there used to be
  // major bugs in this function due to use of auto rather than
  // ReturnType, so I'd prefer to avoid auto completely here.
  ReturnType z_inner_cubical_face =  // NOLINT
      make_with_value<ReturnType>(xi, radius_of_other_surface_ / sqrt(3.0));
  const ReturnType x_unit_spherical_face = tan(M_PI_4 * xi);
  const ReturnType y_unit_spherical_face = tan(M_PI_4 * eta);
  const ReturnType z_spherical_faces_common_factor =
      make_with_value<ReturnType>(xi, 1.) /
      sqrt(1.0 + square(x_unit_spherical_face) + square(y_unit_spherical_face));
  const ReturnType z_inner_spherical_face =
      radius_of_other_surface_ * z_spherical_faces_common_factor;

  // If sphericity_of_other_surface_==0, the inner_cubical_face variables
  // correspond to the
  // face of a cube.
  // If sphericity_of_other_surface_!=0, the inner_cubical_face variables have
  // the
  // spherical_face portions added to them to create a rounded-out cube face.
  // At sphericity_of_other_surface_==1, the face becomes spherical.

  x_inner_cubical_face +=
      sphericity_of_other_surface_ *
      (z_inner_spherical_face * x_unit_spherical_face - x_inner_cubical_face);
  y_inner_cubical_face +=
      sphericity_of_other_surface_ *
      (z_inner_spherical_face * y_unit_spherical_face - y_inner_cubical_face);
  z_inner_cubical_face += sphericity_of_other_surface_ *
                          (z_inner_spherical_face - z_inner_cubical_face);

  const ReturnType z_outer_spherical_face =
      radius_of_spherical_surface_ * z_spherical_faces_common_factor;
  const ReturnType x_outer_spherical_face =
      z_outer_spherical_face * x_unit_spherical_face;
  const ReturnType y_outer_spherical_face =
      z_outer_spherical_face * y_unit_spherical_face;
  const ReturnType physical_x =
      x_inner_cubical_face +
      (x_outer_spherical_face - x_inner_cubical_face) * (zeta + 1) * 0.5;
  const ReturnType physical_y =
      y_inner_cubical_face +
      (y_outer_spherical_face - y_inner_cubical_face) * (zeta + 1) * 0.5;
  const ReturnType physical_z =
      z_inner_cubical_face +
      (z_outer_spherical_face - z_inner_cubical_face) * (zeta + 1) * 0.5;

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

// LCOV_EXCL_START
// Currently unimplemented, hence the noreturn attribute
template <typename T>
[[noreturn]] std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 3>
Wedge3D::inverse(const std::array<T, 3>& /*x*/) const noexcept {
  ERROR("Inverse map is unimplemented for Wedge3D");
}
// LCOV_EXCL_STOP

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Wedge3D::jacobian(const std::array<T, 3>& x) const noexcept {
  using ReturnType = std::decay_t<tt::remove_reference_wrapper_t<T>>;

  const ReturnType& xi = x[0];
  const ReturnType& eta = x[1];
  const ReturnType& zeta = x[2];
  const double ratio_of_radii =
      radius_of_spherical_surface_ / radius_of_other_surface_;
  const ReturnType x_equiangular_secant =
      make_with_value<ReturnType>(xi, 1.0) / cos(M_PI_4 * xi);
  const ReturnType y_equiangular_secant =
      make_with_value<ReturnType>(xi, 1.0) / cos(M_PI_4 * eta);
  const ReturnType x_equiangular_tangent = tan(M_PI_4 * xi);
  const ReturnType y_equiangular_tangent = tan(M_PI_4 * eta);

  const ReturnType sy_square = square(y_equiangular_secant);
  const ReturnType sx_square = square(x_equiangular_secant);
  const ReturnType scaling_and_sphericity = M_PI *
      (ratio_of_radii * (1 + zeta) + sphericity_of_other_surface_ * (1 - zeta));
  const ReturnType denominator_dzeta =
      (2.0 * sqrt(sx_square + square(y_equiangular_tangent)));
  const ReturnType denominator_other = pow<3>(denominator_dzeta);
  const ReturnType nonsphericity = (-1.0 + sphericity_of_other_surface_) * 0.5 *
                                   denominator_dzeta / sqrt(3.);
  const ReturnType trigonometric_nonsphericity =
      nonsphericity * (-1.0 + zeta) *
      (3 + cos(M_PI_2 * xi) + 2 * cos(M_PI_2 * eta) * square(sin(M_PI_4 * xi)));

  Tensor<ReturnType, tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
      jacobian_matrix{};

  auto &dx_dxi = jacobian_matrix.template get<0, 0>(),
       &dx_deta = jacobian_matrix.template get<0, 1>(),
       &dx_dzeta = jacobian_matrix.template get<0, 2>(),
       &dy_dxi = jacobian_matrix.template get<1, 0>(),
       &dy_deta = jacobian_matrix.template get<1, 1>(),
       &dy_dzeta = jacobian_matrix.template get<1, 2>(),
       &dz_dxi = jacobian_matrix.template get<2, 0>(),
       &dz_deta = jacobian_matrix.template get<2, 1>(),
       &dz_dzeta = jacobian_matrix.template get<2, 2>();

  dx_dxi = sy_square * sx_square *
           (scaling_and_sphericity + trigonometric_nonsphericity) /
           denominator_other;
  dx_deta = -scaling_and_sphericity * sy_square * y_equiangular_tangent *
            x_equiangular_tangent / denominator_other;
  dx_dzeta = (nonsphericity * xi +
              (-sphericity_of_other_surface_ + ratio_of_radii) *
                  x_equiangular_tangent) /
             denominator_dzeta;
  dy_dxi = -scaling_and_sphericity * sx_square * y_equiangular_tangent *
           x_equiangular_tangent / denominator_other;
  dy_deta = sy_square * sx_square *
            (scaling_and_sphericity + trigonometric_nonsphericity) /
            denominator_other;
  dy_dzeta = (nonsphericity * eta +
              (-sphericity_of_other_surface_ + ratio_of_radii) *
                  y_equiangular_tangent) /
             denominator_dzeta;
  dz_dxi = -scaling_and_sphericity * sx_square * x_equiangular_tangent /
           denominator_other;
  dz_deta = -scaling_and_sphericity * sy_square * y_equiangular_tangent /
            denominator_other;
  dz_dzeta = (nonsphericity - sphericity_of_other_surface_ + ratio_of_radii) /
             denominator_dzeta;

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

  const double scaled_radius_of_other_surface = radius_of_other_surface_;

  // multiply jacobian_matrix by scaling factor to get jacobian.
  std::transform(jacobian_matrix.begin(), jacobian_matrix.end(),
                 jacobian_matrix.begin(),
                 [scaled_radius_of_other_surface](const T& jacobian_element) {
                   return jacobian_element * scaled_radius_of_other_surface;
                 });
  return jacobian_matrix;
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Wedge3D::inv_jacobian(const std::array<T, 3>& xi) const noexcept {
  return determinant_and_inverse(jacobian(xi)).second;
}

void Wedge3D::pup(PUP::er& p) {
  p | radius_of_other_surface_;
  p | radius_of_spherical_surface_;
  p | direction_of_wedge_;
  p | sphericity_of_other_surface_;
}

bool operator==(const Wedge3D& lhs, const Wedge3D& rhs) noexcept {
  return lhs.radius_of_other_surface_ == rhs.radius_of_other_surface_ and
         lhs.radius_of_spherical_surface_ ==
             rhs.radius_of_spherical_surface_ and
         lhs.direction_of_wedge_ == rhs.direction_of_wedge_ and
         lhs.sphericity_of_other_surface_ == rhs.sphericity_of_other_surface_;
}

bool operator!=(const Wedge3D& lhs, const Wedge3D& rhs) noexcept {
  return not(lhs == rhs);
}

// Explicit instantiations
template std::array<double, 3> Wedge3D::operator()(
    const std::array<std::reference_wrapper<const double>, 3>& /*xi*/) const
    noexcept;
template std::array<double, 3> Wedge3D::operator()(
    const std::array<double, 3>& /*xi*/) const noexcept;
template std::array<DataVector, 3> Wedge3D::operator()(
    const std::array<std::reference_wrapper<const DataVector>, 3>& /*xi*/) const
    noexcept;
template std::array<DataVector, 3> Wedge3D::operator()(
    const std::array<DataVector, 3>& /*xi*/) const noexcept;

template std::array<double, 3> Wedge3D::inverse(
    const std::array<std::reference_wrapper<const double>, 3>& /*xi*/) const
    noexcept;
template std::array<double, 3> Wedge3D::inverse(
    const std::array<double, 3>& /*xi*/) const noexcept;
template std::array<DataVector, 3> Wedge3D::inverse(
    const std::array<std::reference_wrapper<const DataVector>, 3>& /*xi*/) const
    noexcept;
template std::array<DataVector, 3> Wedge3D::inverse(
    const std::array<DataVector, 3>& /*xi*/) const noexcept;

template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Wedge3D::jacobian(
    const std::array<std::reference_wrapper<const double>, 3>& /*xi*/) const
    noexcept;
template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Wedge3D::jacobian(const std::array<double, 3>& /*xi*/) const noexcept;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Wedge3D::jacobian(
    const std::array<std::reference_wrapper<const DataVector>, 3>& /*xi*/) const
    noexcept;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Wedge3D::jacobian(const std::array<DataVector, 3>& /*xi*/) const noexcept;

template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Wedge3D::inv_jacobian(
    const std::array<std::reference_wrapper<const double>, 3>& /*xi*/) const
    noexcept;
template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Wedge3D::inv_jacobian(const std::array<double, 3>& /*xi*/) const noexcept;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Wedge3D::inv_jacobian(
    const std::array<std::reference_wrapper<const DataVector>, 3>& /*xi*/) const
    noexcept;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
Wedge3D::inv_jacobian(const std::array<DataVector, 3>& /*xi*/) const noexcept;
}  // namespace CoordinateMaps
