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
                 const bool with_tan_map) noexcept
    : radius_of_other_surface_(radius_of_other_surface),
      radius_of_spherical_surface_(radius_of_spherical_surface),
      direction_of_wedge_(direction_of_wedge),
      sphericity_of_other_surface_(sphericity_of_other_surface),
      with_tan_map_(with_tan_map) {
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

  const bool with_tan_map = with_tan_map_;
  const double r_inner = radius_of_other_surface_;
  const double r_outer = radius_of_spherical_surface_;
  const double sphericity = sphericity_of_other_surface_;
  const ReturnType& blending_factor =
      (1 - zeta) * sphericity * r_inner + (1 + zeta) * r_outer;
  const ReturnType& cap_xi = with_tan_map ? tan(M_PI_4 * xi) : xi;
  const ReturnType& cap_eta = with_tan_map ? tan(M_PI_4 * eta) : eta;
  const ReturnType& one_over_rho = make_with_value<ReturnType>(xi, 1.) /
                                   sqrt(1.0 + square(cap_xi) + square(cap_eta));
  const ReturnType& physical_x =
      0.5 * (1 - zeta) * (1 - sphericity) * (r_inner / sqrt(3.0)) * xi +
      0.5 * blending_factor * cap_xi * one_over_rho;
  const ReturnType& physical_y =
      0.5 * (1 - zeta) * (1 - sphericity) * (r_inner / sqrt(3.0)) * eta +
      0.5 * blending_factor * cap_eta * one_over_rho;
  const ReturnType& physical_z =
      0.5 * (1 - zeta) * (1 - sphericity) * (r_inner / sqrt(3.0)) +
      0.5 * blending_factor * one_over_rho;

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

  const bool with_tan_map = with_tan_map_;
  const double r_inner = radius_of_other_surface_;
  const double r_outer = radius_of_spherical_surface_;
  const double sphericity = sphericity_of_other_surface_;
  const ReturnType& blending_factor =
      (1 - zeta) * sphericity * r_inner + (1 + zeta) * r_outer;
  const double blending_rate = -sphericity * r_inner + r_outer;
  const ReturnType& cap_xi = with_tan_map ? tan(M_PI_4 * xi) : xi;
  const ReturnType& cap_eta = with_tan_map ? tan(M_PI_4 * eta) : eta;
  const ReturnType& cap_xi_deriv = with_tan_map
                                       ? M_PI_4 * (1 + square(cap_xi))
                                       : make_with_value<ReturnType>(xi, 1.);
  const ReturnType& cap_eta_deriv = with_tan_map
                                        ? M_PI_4 * (1 + square(cap_eta))
                                        : make_with_value<ReturnType>(xi, 1.);
  const ReturnType& one_over_rho = make_with_value<ReturnType>(xi, 1.) /
                                   sqrt(1.0 + square(cap_xi) + square(cap_eta));

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

  dz_dxi =
      -(0.5 * blending_factor * cap_xi * cap_xi_deriv * pow<3>(one_over_rho));
  dz_deta =
      -(0.5 * blending_factor * cap_eta * cap_eta_deriv * pow<3>(one_over_rho));
  dz_dzeta = r_inner * (sphericity - 1.0) / (2 * sqrt(3.0)) +
             0.5 * blending_rate * one_over_rho;

  dx_dxi = r_inner * (1 - sphericity) * (1 - zeta) / (2 * sqrt(3.0)) +
           0.5 * blending_factor * (1 + square(cap_eta)) * cap_xi_deriv *
               pow<3>(one_over_rho);

  dx_deta = cap_xi * dz_deta;
  dx_dzeta = r_inner * (sphericity - 1) * xi / (2 * sqrt(3.0)) +
             0.5 * blending_rate * cap_xi * one_over_rho;
  dy_dxi = cap_eta * dz_dxi;
  dy_deta = dx_dxi;
  dy_dzeta = r_inner * (sphericity - 1) * eta / (2 * sqrt(3.0)) +
             0.5 * blending_rate * cap_eta * one_over_rho;

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
