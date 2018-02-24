// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Wedge3D.hpp"

#include <pup.h>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/OrientationMap.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace CoordinateMaps {

Wedge3D::Wedge3D(const double radius_of_other_surface,
                 const double radius_of_spherical_surface,
                 const OrientationMap<3> orientation_of_wedge,
                 const double sphericity_of_other_surface,
                 const bool with_equiangular_map,
                 const WedgeHalves halves_to_use) noexcept
    : radius_of_other_surface_(radius_of_other_surface),
      radius_of_spherical_surface_(radius_of_spherical_surface),
      orientation_of_wedge_(orientation_of_wedge),
      sphericity_of_other_surface_(sphericity_of_other_surface),
      with_equiangular_map_(with_equiangular_map),
      halves_to_use_(halves_to_use) {
  ASSERT(radius_of_other_surface > 0,
         "The radius of the other surface must be greater than zero.");
  ASSERT(radius_of_spherical_surface > 0,
         "The radius of the spherical surface must be greater than zero.");
  ASSERT(sphericity_of_other_surface >= 0 and sphericity_of_other_surface <= 1,
         "Sphericity of other surface must be between 0 and 1");
  ASSERT(radius_of_other_surface < radius_of_spherical_surface or
             (1. + sphericity_of_other_surface * (sqrt(3.) - 1.)) *
                     radius_of_other_surface >
                 sqrt(3.) * radius_of_spherical_surface,
         "For the value of the given radii and sphericity, the spherical "
         "surface intersects with the other surface");
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> Wedge3D::operator()(
    const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  ReturnType xi = source_coords[0];
  if (halves_to_use_ == WedgeHalves::UpperOnly) {
    xi += 1.0;
    xi *= 0.5;
  }
  if (halves_to_use_ == WedgeHalves::LowerOnly) {
    xi -= 1.0;
    xi *= 0.5;
  }
  const ReturnType& eta = source_coords[1];
  const ReturnType& zeta = source_coords[2];

  const ReturnType& cap_xi = with_equiangular_map_ ? tan(M_PI_4 * xi) : xi;
  const ReturnType& cap_eta = with_equiangular_map_ ? tan(M_PI_4 * eta) : eta;

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

  std::array<ReturnType, 3> physical_coords{
      {std::move(physical_x), std::move(physical_y), std::move(physical_z)}};
  return discrete_rotation(orientation_of_wedge_, std::move(physical_coords));
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> Wedge3D::inverse(
    const std::array<T, 3>& target_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  const std::array<ReturnType, 3> physical_coords =
      discrete_rotation(orientation_of_wedge_.inverse_map(), target_coords);
  const ReturnType& physical_x = physical_coords[0];
  const ReturnType& physical_y = physical_coords[1];
  const ReturnType& physical_z = physical_coords[2];

  const ReturnType cap_xi = physical_x / physical_z;
  const ReturnType cap_eta = physical_y / physical_z;
  const ReturnType one_over_rho =
      1.0 / sqrt(1.0 + square(cap_xi) + square(cap_eta));
  const ReturnType common_factor_one =
      0.5 * radius_of_other_surface_ *
      ((1.0 - sphericity_of_other_surface_) / sqrt(3.0) +
       sphericity_of_other_surface_ * one_over_rho);
  const ReturnType common_factor_two =
      0.5 * radius_of_spherical_surface_ * one_over_rho;
  ReturnType zeta = (physical_z - (common_factor_two + common_factor_one)) /
                    (common_factor_two - common_factor_one);
  ReturnType xi =
      with_equiangular_map_ ? atan(cap_xi) / M_PI_4 : std::move(cap_xi);
  ReturnType eta =
      with_equiangular_map_ ? atan(cap_eta) / M_PI_4 : std::move(cap_eta);
  if (halves_to_use_ == WedgeHalves::UpperOnly) {
    xi *= 2.0;
    xi -= 1.0;
  }
  if (halves_to_use_ == WedgeHalves::LowerOnly) {
    xi *= 2.0;
    xi += 1.0;
  }
  return std::array<ReturnType, 3>{
      {std::move(xi), std::move(eta), std::move(zeta)}};
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> Wedge3D::jacobian(
    const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  ReturnType xi = source_coords[0];
  if (halves_to_use_ == WedgeHalves::UpperOnly) {
    xi += 1.0;
    xi *= 0.5;
  }
  if (halves_to_use_ == WedgeHalves::LowerOnly) {
    xi -= 1.0;
    xi *= 0.5;
  }
  const ReturnType& eta = source_coords[1];
  const ReturnType& zeta = source_coords[2];

  const ReturnType& cap_xi = with_equiangular_map_ ? tan(M_PI_4 * xi) : xi;
  const ReturnType& cap_eta = with_equiangular_map_ ? tan(M_PI_4 * eta) : eta;

  const ReturnType cap_xi_deriv = with_equiangular_map_
                                      ? M_PI_4 * (1.0 + square(cap_xi))
                                      : make_with_value<ReturnType>(xi, 1.0);
  const ReturnType cap_eta_deriv = with_equiangular_map_
                                       ? M_PI_4 * (1.0 + square(cap_eta))
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

  auto jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);

  ReturnType dz_dxi =
      -second_blending_factor_over_rho_cubed * cap_xi * cap_xi_deriv;
  ReturnType dx_dxi =
      (first_blending_factor +
       second_blending_factor_over_rho_cubed * (1.0 + square(cap_eta))) *
      cap_xi_deriv;
  ReturnType dy_dxi = dz_dxi * cap_eta;

  // Implement Scalings:
  if (halves_to_use_ != WedgeHalves::Both) {
    dz_dxi *= 0.5;
    dy_dxi *= 0.5;
    dx_dxi *= 0.5;
  }

  std::array<ReturnType, 3> dX_dlogical = discrete_rotation(
      orientation_of_wedge_,
      std::array<ReturnType, 3>{
          {std::move(dx_dxi), std::move(dy_dxi), std::move(dz_dxi)}});

  get<0, 0>(jacobian_matrix) = dX_dlogical[0];
  get<1, 0>(jacobian_matrix) = dX_dlogical[1];
  get<2, 0>(jacobian_matrix) = dX_dlogical[2];

  // reuse allocation, compute next subset of Jacobian
  dX_dlogical[2] =  // dz_deta
      -second_blending_factor_over_rho_cubed * cap_eta * cap_eta_deriv;
  dX_dlogical[0] =  // dx_deta
      dX_dlogical[2] * cap_xi;
  dX_dlogical[1] =  // dy_deta
      (first_blending_factor +
       second_blending_factor_over_rho_cubed * (1.0 + square(cap_xi))) *
      cap_eta_deriv;

  dX_dlogical =
      discrete_rotation(orientation_of_wedge_, std::move(dX_dlogical));

  get<0, 1>(jacobian_matrix) = dX_dlogical[0];
  get<1, 1>(jacobian_matrix) = dX_dlogical[1];
  get<2, 1>(jacobian_matrix) = dX_dlogical[2];

  dX_dlogical[2] =  // dz_dzeta
      first_blending_rate + second_blending_rate_over_rho;
  dX_dlogical[0] = dX_dlogical[2] * cap_xi;   // dx_dzeta
  dX_dlogical[1] = dX_dlogical[2] * cap_eta;  // dy_dzeta

  dX_dlogical =
      discrete_rotation(orientation_of_wedge_, std::move(dX_dlogical));

  get<0, 2>(jacobian_matrix) = dX_dlogical[0];
  get<1, 2>(jacobian_matrix) = dX_dlogical[1];
  get<2, 2>(jacobian_matrix) = dX_dlogical[2];
  return jacobian_matrix;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> Wedge3D::inv_jacobian(
    const std::array<T, 3>& source_coords) const noexcept {
  return determinant_and_inverse(jacobian(source_coords)).second;
}

void Wedge3D::pup(PUP::er& p) noexcept {
  p | radius_of_other_surface_;
  p | radius_of_spherical_surface_;
  p | orientation_of_wedge_;
  p | sphericity_of_other_surface_;
  p | with_equiangular_map_;
  p | halves_to_use_;
}

bool operator==(const Wedge3D& lhs, const Wedge3D& rhs) noexcept {
  return lhs.radius_of_other_surface_ == rhs.radius_of_other_surface_ and
         lhs.radius_of_spherical_surface_ ==
             rhs.radius_of_spherical_surface_ and
         lhs.orientation_of_wedge_ == rhs.orientation_of_wedge_ and
         lhs.sphericity_of_other_surface_ ==
             rhs.sphericity_of_other_surface_ and
         lhs.with_equiangular_map_ == rhs.with_equiangular_map_ and
         lhs.halves_to_use_ == rhs.halves_to_use_;
}

bool operator!=(const Wedge3D& lhs, const Wedge3D& rhs) noexcept {
  return not(lhs == rhs);
}

// Explicit instantiations
/// \cond
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3> Wedge3D::      \
  operator()(const std::array<DTYPE(data), 3>& source_coords) const noexcept; \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>                \
  Wedge3D::inverse(const std::array<DTYPE(data), 3>& target_coords)           \
      const noexcept;                                                         \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>  \
  Wedge3D::jacobian(const std::array<DTYPE(data), 3>& source_coords)          \
      const noexcept;                                                         \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>  \
  Wedge3D::inv_jacobian(const std::array<DTYPE(data), 3>& source_coords)      \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))

#undef DTYPE
#undef INSTANTIATE
/// \endcond
}  // namespace CoordinateMaps
