// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/FocallyLiftedFlatEndcap.hpp"

#include <cmath>
#include <limits>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace domain::CoordinateMaps::FocallyLiftedInnerMaps {

FlatEndcap::FlatEndcap(const std::array<double, 3>& center, double radius)
    : center_(center), radius_([&radius]() {
        ASSERT(
            not equal_within_roundoff(radius, 0.0),
            "Cannot have zero radius.  Note that this ASSERT implicitly "
            "assumes that the radius has a scale of roughly unity.  Therefore, "
            "this ASSERT may trigger in the case where we intentionally want "
            "an entire domain that is very small.  If we really want small "
            "domains, then this ASSERT should be modified.");
        return radius;
      }()) {}

template <typename T>
void FlatEndcap::forward_map(
    const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*>
        target_coords,
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& xbar = source_coords[0];
  const ReturnType& ybar = source_coords[1];

  if constexpr (not std::is_same_v<ReturnType, double>) {
    (*target_coords)[2].destructive_resize(
        static_cast<ReturnType>(source_coords[0]).size());
  }

  (*target_coords)[0] = radius_ * xbar + center_[0];
  (*target_coords)[1] = radius_ * ybar + center_[1];
  (*target_coords)[2] = center_[2];
}

template <typename T>
void FlatEndcap::jacobian(
    const gsl::not_null<
        tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>*>
        jacobian_out,
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  if constexpr (not std::is_same_v<ReturnType, double>) {
    destructive_resize_components(
        jacobian_out, static_cast<ReturnType>(source_coords[0]).size());
  }
  // Most of the jacobian components are zero.
  for (auto& jac_component : *jacobian_out) {
    jac_component = 0.0;
  }

  // dx/dxbar
  get<0, 0>(*jacobian_out) = radius_;
  // dy/dybar
  get<1, 1>(*jacobian_out) = radius_;
}

template <typename T>
void FlatEndcap::inv_jacobian(
    const gsl::not_null<
        tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>*>
        inv_jacobian_out,
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  if constexpr (not std::is_same_v<ReturnType, double>) {
    destructive_resize_components(
        inv_jacobian_out, static_cast<ReturnType>(source_coords[0]).size());
  }
  // Most of the inverse jacobian components are zero.
  for (auto& jac_component : *inv_jacobian_out) {
    jac_component = 0.0;
  }

  // dxbar/dx
  get<0, 0>(*inv_jacobian_out) = 1.0 / radius_;
  // dybar/dy
  get<1, 1>(*inv_jacobian_out) = 1.0 / radius_;
}

std::optional<std::array<double, 3>> FlatEndcap::inverse(
    const std::array<double, 3>& target_coords, const double sigma_in) const {
  const double xbar = (target_coords[0] - center_[0]) / radius_;
  const double ybar = (target_coords[1] - center_[1]) / radius_;
  const double zbar = 2.0 * sigma_in - 1.0;

  if (abs(zbar) > 1.0 and not equal_within_roundoff(abs(zbar), 1.0)) {
    return {};
  }
  const double rho_squared = square(xbar) + square(ybar);
  if (rho_squared > 1.0 and not equal_within_roundoff(rho_squared, 1.0)) {
    return {};
  }

  return std::array<double, 3>{{xbar, ybar, zbar}};
}

template <typename T>
void FlatEndcap::sigma(
    const gsl::not_null<tt::remove_cvref_wrap_t<T>*> sigma_out,
    const std::array<T, 3>& source_coords) const {
  *sigma_out = 0.5 * (source_coords[2] + 1.0);
}

template <typename T>
void FlatEndcap::deriv_sigma(
    const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*>
        deriv_sigma_out,
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  if constexpr (not std::is_same_v<ReturnType, double>) {
    destructive_resize_components(
        deriv_sigma_out, static_cast<ReturnType>(source_coords[0]).size());
  }
  (*deriv_sigma_out)[0] = 0.0;
  (*deriv_sigma_out)[1] = 0.0;
  (*deriv_sigma_out)[2] = 0.5;
}

template <typename T>
void FlatEndcap::dxbar_dsigma(
    const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*>
        dxbar_dsigma_out,
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  if constexpr (not std::is_same_v<ReturnType, double>) {
    destructive_resize_components(
        dxbar_dsigma_out, static_cast<ReturnType>(source_coords[0]).size());
  }
  (*dxbar_dsigma_out)[0] = 0.0;
  (*dxbar_dsigma_out)[1] = 0.0;
  (*dxbar_dsigma_out)[2] = 2.0;
}

std::optional<double> FlatEndcap::lambda_tilde(
    const std::array<double, 3>& parent_mapped_target_coords,
    const std::array<double, 3>& projection_point,
    const bool /*source_is_between_focus_and_target*/) const {
  const double result = (center_[2] - projection_point[2]) /
                        (parent_mapped_target_coords[2] - projection_point[2]);
  if (result < 1.0) {
    return {};
  }
  return result;
}

template <typename T>
void FlatEndcap::deriv_lambda_tilde(
    const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*>
        deriv_lambda_tilde_out,
    const std::array<T, 3>& target_coords, const T& lambda_tilde,
    const std::array<double, 3>& projection_point) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  if constexpr (not std::is_same_v<ReturnType, double>) {
    destructive_resize_components(
        deriv_lambda_tilde_out,
        static_cast<ReturnType>(target_coords[0]).size());
  }

  (*deriv_lambda_tilde_out)[0] = 0.0;
  (*deriv_lambda_tilde_out)[1] = 0.0;
  (*deriv_lambda_tilde_out)[2] =
      -square(lambda_tilde) / (center_[2] - projection_point[2]);
}

void FlatEndcap::pup(PUP::er& p) {
  p | center_;
  p | radius_;
}

bool operator==(const FlatEndcap& lhs, const FlatEndcap& rhs) {
  return lhs.center_ == rhs.center_ and lhs.radius_ == rhs.radius_;
}

bool operator!=(const FlatEndcap& lhs, const FlatEndcap& rhs) {
  return not(lhs == rhs);
}
// Explicit instantiations
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template void FlatEndcap::forward_map(                                      \
      const gsl::not_null<                                                    \
          std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>*>               \
          target_coords,                                                      \
      const std::array<DTYPE(data), 3>& source_coords) const;                 \
  template void FlatEndcap::jacobian(                                         \
      const gsl::not_null<                                                    \
          tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>*> \
          jacobian_out,                                                       \
      const std::array<DTYPE(data), 3>& source_coords) const;                 \
  template void FlatEndcap::inv_jacobian(                                     \
      const gsl::not_null<                                                    \
          tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>*> \
          inv_jacobian_out,                                                   \
      const std::array<DTYPE(data), 3>& source_coords) const;                 \
  template void FlatEndcap::sigma(                                            \
      const gsl::not_null<tt::remove_cvref_wrap_t<DTYPE(data)>*> sigma_out,   \
      const std::array<DTYPE(data), 3>& source_coords) const;                 \
  template void FlatEndcap::deriv_sigma(                                      \
      const gsl::not_null<                                                    \
          std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>*>               \
          deriv_sigma_out,                                                    \
      const std::array<DTYPE(data), 3>& source_coords) const;                 \
  template void FlatEndcap::dxbar_dsigma(                                     \
      const gsl::not_null<                                                    \
          std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>*>               \
          dxbar_dsigma_out,                                                   \
      const std::array<DTYPE(data), 3>& source_coords) const;                 \
  template void FlatEndcap::deriv_lambda_tilde(                               \
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
