// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/FocallyLiftedEndcap.hpp"

#include <cmath>
#include <limits>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CylindricalEndcapHelpers.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedMapHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/PupStlCpp11.hpp"

namespace domain::CoordinateMaps::FocallyLiftedInnerMaps {

Endcap::Endcap(const std::array<double, 3>& center, const double radius,
               const double z_plane)
    : center_(center),
      radius_([&radius]() {
        // The equal_within_roundoff below has an implicit scale of 1,
        // so the ASSERT may trigger in the case where we really
        // want an entire domain that is very small.
        ASSERT(not equal_within_roundoff(radius, 0.0),
               "Cannot have zero radius");
        return radius;
      }()),
      theta_max_([&center, &radius, &z_plane]() {
        const double cos_theta_max = (z_plane - center[2]) / radius;
        ASSERT(abs(cos_theta_max) < 1.0,
               "Plane must intersect sphere, and at more than one point");
        return acos(cos_theta_max);
      }()) {}

template <typename T>
void Endcap::forward_map(
    const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*>
        target_coords,
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& xbar = source_coords[0];
  const ReturnType& ybar = source_coords[1];
  ReturnType& x = (*target_coords)[0];
  ReturnType& y = (*target_coords)[1];
  ReturnType& z = (*target_coords)[2];
  // Use z and y as temporary storage to avoid allocations,
  // before setting them to their actual values.
  z = sqrt(square(xbar) + square(ybar));
  y = cylindrical_endcap_helpers::sin_ax_over_x(z, theta_max_) * radius_;
  x = y * xbar + center_[0];
  y = y * ybar + center_[1];
  z = radius_ * cos(z * theta_max_) + center_[2];
}

template <typename T>
void Endcap::jacobian(const gsl::not_null<tnsr::Ij<tt::remove_cvref_wrap_t<T>,
                                                   3, Frame::NoFrame>*>
                          jacobian_out,
                      const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& xbar = source_coords[0];
  const ReturnType& ybar = source_coords[1];

  if constexpr (not std::is_same<ReturnType, double>::value) {
    destructive_resize_components(
        jacobian_out, static_cast<ReturnType>(source_coords[0]).size());
  }
  // Most of the jacobian components are zero.
  for (auto& jac_component : *jacobian_out) {
    jac_component = 0.0;
  }

  // Use parts of Jacobian as temp storage to reduce allocations.
  get<1, 1>(*jacobian_out) = sqrt(square(xbar) + square(ybar));
  get<1, 0>(*jacobian_out) =
      radius_ * cylindrical_endcap_helpers::sin_ax_over_x(
                    get<1, 1>(*jacobian_out), theta_max_);
  // 1/rhobar d/d(rhobar) [(sin (rhobar theta_max)/rhobar)]
  get<0, 1>(*jacobian_out) =
      radius_ * cylindrical_endcap_helpers::one_over_x_d_sin_ax_over_x(
                    get<1, 1>(*jacobian_out), theta_max_);

  // dy/dybar
  get<1, 1>(*jacobian_out) =
      get<0, 1>(*jacobian_out) * square(ybar) + get<1, 0>(*jacobian_out);
  // dx/dxbar
  get<0, 0>(*jacobian_out) =
      get<0, 1>(*jacobian_out) * square(xbar) + get<1, 0>(*jacobian_out);

  // Still a temporary variable here.
  get<1, 0>(*jacobian_out) *= -theta_max_;

  // dz/dxbar
  get<2, 0>(*jacobian_out) = get<1, 0>(*jacobian_out) * xbar;
  // dz/dybar
  get<2, 1>(*jacobian_out) = get<1, 0>(*jacobian_out) * ybar;

  // dx/dybar
  get<0, 1>(*jacobian_out) *= xbar * ybar;
  // dy/dxbar
  get<1, 0>(*jacobian_out) = get<0, 1>(*jacobian_out);
}

template <typename T>
void Endcap::inv_jacobian(
    const gsl::not_null<
        tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>*>
        inv_jacobian_out,
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& xbar = source_coords[0];
  const ReturnType& ybar = source_coords[1];

  if constexpr (not std::is_same<ReturnType, double>::value) {
    destructive_resize_components(
        inv_jacobian_out, static_cast<ReturnType>(source_coords[0]).size());
  }
  // Most of the inverse jacobian components are zero.
  for (auto& jac_component : *inv_jacobian_out) {
    jac_component = 0.0;
  }

  // Use parts of Jacobian as temp storage to reduce allocations.
  // We comment temporary quantities to make the code more understandable.

  // rhobar, Eq. 7 in the documentation.
  get<1, 0>(*inv_jacobian_out) = sqrt(square(xbar) + square(ybar));

  // q = sin(rhobar theta)/rhobar as defined in the documentation, Eq. 17.
  get<0, 1>(*inv_jacobian_out) = cylindrical_endcap_helpers::sin_ax_over_x(
      get<1, 0>(*inv_jacobian_out), theta_max_);

  // 1/rhobar dq/d(rhobar)
  get<1, 1>(*inv_jacobian_out) =
      cylindrical_endcap_helpers::one_over_x_d_sin_ax_over_x(
          get<1, 0>(*inv_jacobian_out), theta_max_);

  // Right-hand side of Eq. 23, without the factor of
  // xbar ybar/rq or the minus sign.
  get<1, 0>(*inv_jacobian_out) =
      get<1, 1>(*inv_jacobian_out) /
      (get<0, 1>(*inv_jacobian_out) +
       square(get<1, 0>(*inv_jacobian_out)) * get<1, 1>(*inv_jacobian_out));

  // 1/(r q), i.e. first term in Eq. 22.
  get<0, 0>(*inv_jacobian_out) = 1.0 / (get<0, 1>(*inv_jacobian_out) * radius_);

  // Right-hand side of Eq. 23, without the factor of
  // xbar ybar or the minus sign.
  get<1, 0>(*inv_jacobian_out) *= get<0, 0>(*inv_jacobian_out);

  // dybar/dy
  get<1, 1>(*inv_jacobian_out) = get<0, 0>(*inv_jacobian_out) -
                                 square(ybar) * get<1, 0>(*inv_jacobian_out);
  // dxbar/dx
  get<0, 0>(*inv_jacobian_out) -= square(xbar) * get<1, 0>(*inv_jacobian_out);
  // dybar/dx
  get<1, 0>(*inv_jacobian_out) *= -xbar * ybar;
  // dxbar/dy
  get<0, 1>(*inv_jacobian_out) = get<1, 0>(*inv_jacobian_out);
}

std::optional<std::array<double, 3>> Endcap::inverse(
    const std::array<double, 3>& target_coords, const double sigma_in) const {
  const double x = target_coords[0] - center_[0];
  const double y = target_coords[1] - center_[1];
  const double z = target_coords[2] - center_[2];

  // Are we in the range of the map?
  // The equal_within_roundoff below has an implicit scale of 1,
  // so the inverse may fail if radius_ is very small on purpose,
  // e.g. if we really want a tiny tiny domain for some reason.
  const double r = sqrt(square(x) + square(y) + square(z));
  if (not equal_within_roundoff(r, radius_)) {
    return std::optional<std::array<double, 3>>{};
  }

  // Compute zbar and check if we are in the range of the map.
  const double zbar = 2.0 * sigma_in - 1.0;
  if (abs(zbar) > 1.0 and not equal_within_roundoff(abs(zbar), 1.0)) {
    return std::optional<std::array<double, 3>>{};
  }

  const double rho = sqrt(square(x) + square(y));
  if (UNLIKELY(rho == 0.0)) {
    // If x and y are zero, so are xbar and ybar,
    // so we are done.
    return std::array<double, 3>{{0.0, 0.0, zbar}};
  }

  // Note: theta_max_ cannot be zero for a nonsingular map.
  const double rhobar = atan2(rho, z) / theta_max_;

  // Check if we are outside the range of the map.
  if (rhobar > 1.0 and not equal_within_roundoff(rhobar, 1.0)) {
    return std::optional<std::array<double, 3>>{};
  }

  const double xbar = x * rhobar / rho;
  const double ybar = y * rhobar / rho;
  return std::array<double, 3>{{xbar, ybar, zbar}};
}

template <typename T>
void Endcap::sigma(const gsl::not_null<tt::remove_cvref_wrap_t<T>*> sigma_out,
                   const std::array<T, 3>& source_coords) const {
  *sigma_out = 0.5 * (source_coords[2] + 1.0);
}

template <typename T>
void Endcap::deriv_sigma(
    const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*>
        deriv_sigma_out,
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  if constexpr (not std::is_same<ReturnType, double>::value) {
    destructive_resize_components(
        deriv_sigma_out, static_cast<ReturnType>(source_coords[0]).size());
  }
  (*deriv_sigma_out)[0] = 0.0;
  (*deriv_sigma_out)[1] = 0.0;
  (*deriv_sigma_out)[2] = 0.5;
}

template <typename T>
void Endcap::dxbar_dsigma(
    const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*>
        dxbar_dsigma_out,
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  if constexpr (not std::is_same<ReturnType, double>::value) {
    destructive_resize_components(
        dxbar_dsigma_out, static_cast<ReturnType>(source_coords[0]).size());
  }
  (*dxbar_dsigma_out)[0] = 0.0;
  (*dxbar_dsigma_out)[1] = 0.0;
  (*dxbar_dsigma_out)[2] = 2.0;
}

std::optional<double> Endcap::lambda_tilde(
    const std::array<double, 3>& parent_mapped_target_coords,
    const std::array<double, 3>& projection_point,
    const bool source_is_between_focus_and_target) const {
  // Try to find lambda_tilde going from target_coords to sphere.
  // If the target surface is outside the sphere (that is,
  // source_is_between_focus_and_target is true), then lambda_tilde should be
  // positive and less than or equal to unity. If the target surface is inside
  // the sphere, then lambda_tilde should be greater than or equal
  // to unity. If there are two such roots, we choose based on where the points
  // are.
  const bool choose_larger_root =
      parent_mapped_target_coords[2] > projection_point[2];
  return FocallyLiftedMapHelpers::try_scale_factor(
      parent_mapped_target_coords, projection_point, center_, radius_,
      choose_larger_root, not source_is_between_focus_and_target);
}

template <typename T>
void Endcap::deriv_lambda_tilde(
    const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*>
        deriv_lambda_tilde_out,
    const std::array<T, 3>& target_coords, const T& lambda_tilde,
    const std::array<double, 3>& projection_point) const {
  FocallyLiftedMapHelpers::d_scale_factor_d_src_point(
      deriv_lambda_tilde_out, target_coords, projection_point, center_,
      lambda_tilde);
}

void Endcap::pup(PUP::er& p) {
  size_t version = 0;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | center_;
    p | radius_;
    p | theta_max_;
  }
}

bool operator==(const Endcap& lhs, const Endcap& rhs) {
  return lhs.center_ == rhs.center_ and lhs.radius_ == rhs.radius_ and
         lhs.theta_max_ == rhs.theta_max_;
}

bool operator!=(const Endcap& lhs, const Endcap& rhs) {
  return not(lhs == rhs);
}
// Explicit instantiations
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template void Endcap::forward_map(                                          \
      const gsl::not_null<                                                    \
          std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>*>               \
          target_coords,                                                      \
      const std::array<DTYPE(data), 3>& source_coords) const;                 \
  template void Endcap::jacobian(                                             \
      const gsl::not_null<                                                    \
          tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>*> \
          jacobian_out,                                                       \
      const std::array<DTYPE(data), 3>& source_coords) const;                 \
  template void Endcap::inv_jacobian(                                         \
      const gsl::not_null<                                                    \
          tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>*> \
          inv_jacobian_out,                                                   \
      const std::array<DTYPE(data), 3>& source_coords) const;                 \
  template void Endcap::sigma(                                                \
      const gsl::not_null<tt::remove_cvref_wrap_t<DTYPE(data)>*> sigma_out,   \
      const std::array<DTYPE(data), 3>& source_coords) const;                 \
  template void Endcap::deriv_sigma(                                          \
      const gsl::not_null<                                                    \
          std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>*>               \
          deriv_sigma_out,                                                    \
      const std::array<DTYPE(data), 3>& source_coords) const;                 \
  template void Endcap::dxbar_dsigma(                                         \
      const gsl::not_null<                                                    \
          std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>*>               \
          dxbar_dsigma_out,                                                   \
      const std::array<DTYPE(data), 3>& source_coords) const;                 \
  template void Endcap::deriv_lambda_tilde(                                   \
      const gsl::not_null<                                                    \
          std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>*>               \
          deriv_lambda_tilde_out,                                             \
      const std::array<DTYPE(data), 3>& target_coords,                        \
      const DTYPE(data) & lambda_tilde,                                       \
      const std::array<double, 3>& projection_point) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))

#undef INSTANTIATE
#undef DTYPE

}  // namespace domain::CoordinateMaps::FocallyLiftedInnerMaps
