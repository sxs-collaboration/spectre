// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/FocallyLiftedSide.hpp"

#include <cmath>
#include <limits>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedMapHelpers.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain::CoordinateMaps::FocallyLiftedInnerMaps {

Side::Side(const std::array<double, 3>& center, const double radius,
           const double z_lower, const double z_upper) noexcept
    : center_(center),
      radius_([&radius]() noexcept {
        ASSERT(
            not equal_within_roundoff(radius, 0.0),
            "Cannot have zero radius.  Note that this ASSERT implicitly "
            "assumes that the radius has a scale of roughly unity.  Therefore, "
            "this ASSERT may trigger in the case where we intentionally want "
            "an entire domain that is very small.  If we really want small "
            "domains, then this ASSERT should be modified.");
        return radius;
      }()),
      theta_min_([&z_upper, &center, &radius]() noexcept {
        ASSERT(abs(z_upper - center[2]) < radius,
               "Upper plane must intersect sphere, and it must do "
               "so at more than one point.");
        return acos((z_upper - center[2]) / radius);
      }()),
      theta_max_([&z_lower, &center, &radius]() noexcept {
        ASSERT(abs(z_lower - center[2]) < radius,
               "Lower plane must intersect sphere, and it must do "
               "so at more than one point.");
        return acos((z_lower - center[2]) / radius);
      }()) {
  // Note that theta decreases with increasing z, which is why
  // theta_min_ above uses z_upper and theta_max_ uses z_lower.
  ASSERT(z_lower < z_upper, "Lower plane must be below upper plane");
}

template <typename T>
void Side::forward_map(
    const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*>
        target_coords,
    const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& xbar = source_coords[0];
  const ReturnType& ybar = source_coords[1];
  const ReturnType& zbar = source_coords[2];
  ReturnType& x = (*target_coords)[0];
  ReturnType& y = (*target_coords)[1];
  ReturnType& z = (*target_coords)[2];
  // Use z and y as temporary storage to avoid allocations,
  // before setting them to their actual values.
  z = theta_max_ + 0.5 * (theta_min_ - theta_max_) * (1.0 + zbar);
  // Note: denominator of the next line is guaranteed != 0 because for
  // this map xbar and ybar are coordinates of points on the *sides*
  // of a right cylinder, i.e. away from the zbar-axis.
  y = radius_ * sin(z) / sqrt(square(xbar) + square(ybar));
  x = y * xbar + center_[0];
  y = y * ybar + center_[1];
  z = radius_ * cos(z) + center_[2];
}

template <typename T>
void Side::jacobian(const gsl::not_null<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3,
                                                 Frame::NoFrame>*>
                        jacobian_out,
                    const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& xbar = source_coords[0];
  const ReturnType& ybar = source_coords[1];
  const ReturnType& zbar = source_coords[2];

  if constexpr (not std::is_same_v<ReturnType, double>) {
    destructive_resize_components(
        jacobian_out, static_cast<ReturnType>(source_coords[0]).size());
  }

  // Use (1,1) (2,2), and (1,2) parts of Jacobian as temp storage to
  // reduce allocations.
  get<1, 1>(*jacobian_out) =
      theta_max_ + 0.5 * (theta_min_ - theta_max_) * (1.0 + zbar);
  get<2, 2>(*jacobian_out) = radius_ * sin(get<1, 1>(*jacobian_out));
  get<1, 2>(*jacobian_out) =
      radius_ * 0.5 * (theta_min_ - theta_max_) * cos(get<1, 1>(*jacobian_out));
  // Denominator of next line is guaranteed to not be zero.
  get<1, 1>(*jacobian_out) = 1.0 / sqrt(square(xbar) + square(ybar));
  get<1, 2>(*jacobian_out) *= get<1, 1>(*jacobian_out);
  get<1, 1>(*jacobian_out) =
      get<2, 2>(*jacobian_out) * cube(get<1, 1>(*jacobian_out));

  // dx/dxbar
  get<0, 0>(*jacobian_out) = get<1, 1>(*jacobian_out) * square(ybar);
  // dx/dybar
  get<0, 1>(*jacobian_out) = -get<1, 1>(*jacobian_out) * xbar * ybar;
  // dx/dzbar
  get<0, 2>(*jacobian_out) = get<1, 2>(*jacobian_out) * xbar;
  // dy/dxbar
  get<1, 0>(*jacobian_out) = -get<1, 1>(*jacobian_out) * xbar * ybar;
  // dy/dybar
  get<1, 1>(*jacobian_out) *= square(xbar);
  // dy/dzbar
  get<1, 2>(*jacobian_out) *= ybar;
  // dz/dzbar
  get<2, 2>(*jacobian_out) *= -0.5 * (theta_min_ - theta_max_);

  // The rest of the components are zero
  get<2, 0>(*jacobian_out) = 0.0;
  get<2, 1>(*jacobian_out) = 0.0;
}

template <typename T>
void Side::inv_jacobian(const gsl::not_null<tnsr::Ij<tt::remove_cvref_wrap_t<T>,
                                                     3, Frame::NoFrame>*>
                            inv_jacobian_out,
                        const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& xbar = source_coords[0];
  const ReturnType& ybar = source_coords[1];
  const ReturnType& zbar = source_coords[2];

  if constexpr (not std::is_same<ReturnType, double>::value) {
    destructive_resize_components(
        inv_jacobian_out, static_cast<ReturnType>(source_coords[0]).size());
  }

  // Use parts of inverse Jacobian as temp storage to reduce allocations.
  const double theta_factor = 0.5 * (theta_min_ - theta_max_);
  get<2, 2>(*inv_jacobian_out) =
      1.0 / (radius_ * sin(theta_max_ + theta_factor * (1.0 + zbar)));
  // Denominator of next line is guaranteed to be nonzero.
  get<1, 1>(*inv_jacobian_out) =
      get<2, 2>(*inv_jacobian_out) / sqrt(square(xbar) + square(ybar));

  // dxbar/dx
  get<0, 0>(*inv_jacobian_out) = square(ybar) * get<1, 1>(*inv_jacobian_out);
  // dxbar/dy
  get<0, 1>(*inv_jacobian_out) = -xbar * ybar * get<1, 1>(*inv_jacobian_out);
  // dybar/dx
  get<1, 0>(*inv_jacobian_out) = get<0, 1>(*inv_jacobian_out);
  // dybar/dy
  get<1, 1>(*inv_jacobian_out) *= square(xbar);
  // dzbar/dz
  get<2, 2>(*inv_jacobian_out) /= -theta_factor;

  // The rest of the components are zero
  get<0, 2>(*inv_jacobian_out) = 0.0;
  get<1, 2>(*inv_jacobian_out) = 0.0;
  get<2, 0>(*inv_jacobian_out) = 0.0;
  get<2, 1>(*inv_jacobian_out) = 0.0;
}

std::optional<std::array<double, 3>> Side::inverse(
    const std::array<double, 3>& target_coords,
    const double sigma_in) const noexcept {
  if ((sigma_in < 0.0 and not equal_within_roundoff(sigma_in, 0.0)) or
      (sigma_in > 1.0 and not equal_within_roundoff(sigma_in, 1.0))) {
    return {};
  }

  const double x = target_coords[0] - center_[0];
  const double y = target_coords[1] - center_[1];
  const double z = target_coords[2] - center_[2];
  const double rho = sqrt(square(x) + square(y));

  const double zbar =
      2.0 * ((acos(z / radius_) - theta_max_) / (theta_min_ - theta_max_)) -
      1.0;
  if ((zbar < -1.0 and not equal_within_roundoff(zbar, -1.0)) or
      (zbar > 1.0 and not equal_within_roundoff(zbar, 1.0))) {
    return {};
  }

  const double xbar = (1.0 + sigma_in) * x / rho;
  const double ybar = (1.0 + sigma_in) * y / rho;
  return std::array<double, 3>{{xbar, ybar, zbar}};
}

template <typename T>
void Side::sigma(const gsl::not_null<tt::remove_cvref_wrap_t<T>*> sigma_out,
                 const std::array<T, 3>& source_coords) const noexcept {
  *sigma_out = sqrt(square(source_coords[0]) + square(source_coords[1])) - 1.0;
}

template <typename T>
void Side::deriv_sigma(
    const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*>
        deriv_sigma_out,
    const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& xbar = source_coords[0];
  const ReturnType& ybar = source_coords[1];
  if constexpr (not std::is_same<ReturnType, double>::value) {
    destructive_resize_components(
        deriv_sigma_out, static_cast<ReturnType>(source_coords[0]).size());
  }

  // Use as temp storage to reduce allocations.
  // Note that denominator is guaranteed to be nonzero.
  (*deriv_sigma_out)[1] = 1.0 / sqrt(square(xbar) + square(ybar));

  (*deriv_sigma_out)[0] = xbar * (*deriv_sigma_out)[1];
  (*deriv_sigma_out)[1] *= ybar;
  (*deriv_sigma_out)[2] = 0.0;
}

template <typename T>
void Side::dxbar_dsigma(
    const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*>
        dxbar_dsigma_out,
    const std::array<T, 3>& source_coords) const noexcept {
  deriv_sigma(dxbar_dsigma_out, source_coords);
}

std::optional<double> Side::lambda_tilde(
    const std::array<double, 3>& parent_mapped_target_coords,
    const std::array<double, 3>& projection_point,
    const bool source_is_between_focus_and_target) const noexcept {
  return FocallyLiftedMapHelpers::try_scale_factor(
      parent_mapped_target_coords, projection_point, center_, radius_,
      source_is_between_focus_and_target,
      not source_is_between_focus_and_target, false);
}

template <typename T>
void Side::deriv_lambda_tilde(
    const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*>
        deriv_lambda_tilde_out,
    const std::array<T, 3>& target_coords, const T& lambda_tilde,
    const std::array<double, 3>& projection_point) const noexcept {
  FocallyLiftedMapHelpers::d_scale_factor_d_src_point(
      deriv_lambda_tilde_out, target_coords, projection_point, center_,
      lambda_tilde);
}

void Side::pup(PUP::er& p) noexcept {
  p | center_;
  p | radius_;
  p | theta_min_;
  p | theta_max_;
}

bool operator==(const Side& lhs, const Side& rhs) noexcept {
  return lhs.center_ == rhs.center_ and lhs.radius_ == rhs.radius_ and
         lhs.theta_min_ == rhs.theta_min_ and lhs.theta_max_ == rhs.theta_max_;
}

bool operator!=(const Side& lhs, const Side& rhs) noexcept {
  return not(lhs == rhs);
}
// Explicit instantiations
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template void Side::forward_map(                                            \
      const gsl::not_null<                                                    \
          std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>*>               \
          target_coords,                                                      \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;        \
  template void Side::jacobian(                                               \
      const gsl::not_null<                                                    \
          tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>*> \
          jacobian_out,                                                       \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;        \
  template void Side::inv_jacobian(                                           \
      const gsl::not_null<                                                    \
          tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>*> \
          inv_jacobian_out,                                                   \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;        \
  template void Side::sigma(                                                  \
      const gsl::not_null<tt::remove_cvref_wrap_t<DTYPE(data)>*> sigma_out,   \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;        \
  template void Side::deriv_sigma(                                            \
      const gsl::not_null<                                                    \
          std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>*>               \
          deriv_sigma_out,                                                    \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;        \
  template void Side::dxbar_dsigma(                                           \
      const gsl::not_null<                                                    \
          std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>*>               \
          dxbar_dsigma_out,                                                   \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;        \
  template void Side::deriv_lambda_tilde(                                     \
      const gsl::not_null<                                                    \
          std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>*>               \
          deriv_lambda_tilde_out,                                             \
      const std::array<DTYPE(data), 3>& target_coords,                        \
      const DTYPE(data) & lambda_tilde,                                       \
      const std::array<double, 3>& projection_point) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))

#undef INSTANTIATE
#undef DTYPE

}  // namespace domain::CoordinateMaps::FocallyLiftedInnerMaps
