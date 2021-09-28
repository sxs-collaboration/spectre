// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/FocallyLiftedFlatSide.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <pup.h>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

namespace domain::CoordinateMaps::FocallyLiftedInnerMaps {

FlatSide::FlatSide(const std::array<double, 3>& center,
                   const double inner_radius, const double outer_radius)
    : center_(center),
      inner_radius_(inner_radius),
      outer_radius_(outer_radius) {
  ASSERT(not equal_within_roundoff(inner_radius, 0.0),
         "Cannot have zero radius.  Note that this ASSERT implicitly "
         "assumes that the radius has a scale of roughly unity.  Therefore, "
         "this ASSERT may trigger in the case where we intentionally want "
         "an entire domain that is very small.  If we really want small "
         "domains, then this ASSERT should be modified.");
  ASSERT(outer_radius > inner_radius,
         "Outer radius should be larger than inner radius. Inner radius: "
             << inner_radius << " outer radius: " << outer_radius);
}

template <typename T>
void FlatSide::forward_map(
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
  y = (inner_radius_ + (outer_radius_ - inner_radius_) * (z - 1.0)) / z;

  x = y * xbar + center_[0];
  y = y * ybar + center_[1];
  z = center_[2];
}

template <typename T>
void FlatSide::jacobian(const gsl::not_null<tnsr::Ij<tt::remove_cvref_wrap_t<T>,
                                                     3, Frame::NoFrame>*>
                            jacobian_out,
                        const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& xbar = source_coords[0];
  const ReturnType& ybar = source_coords[1];

  if constexpr (not std::is_same_v<ReturnType, double>) {
    destructive_resize_components(
        jacobian_out, static_cast<ReturnType>(source_coords[0]).size());
  }

  // Use part of Jacobian as temp storage to reduce allocations.
  get<1, 1>(*jacobian_out) = 1.0 / cube(sqrt(square(xbar) + square(ybar)));

  // dx/dxbar
  get<0, 0>(*jacobian_out) = (2.0 * inner_radius_ - outer_radius_) *
                                 square(ybar) * get<1, 1>(*jacobian_out) +
                             (outer_radius_ - inner_radius_);
  // dx/dybar
  get<0, 1>(*jacobian_out) = (outer_radius_ - 2.0 * inner_radius_) * xbar *
                             ybar * get<1, 1>(*jacobian_out);
  // dy/dxbar
  get<1, 0>(*jacobian_out) = get<0, 1>(*jacobian_out);
  // dy/dybar
  get<1, 1>(*jacobian_out) = (2.0 * inner_radius_ - outer_radius_) *
                                 square(xbar) * get<1, 1>(*jacobian_out) +
                             (outer_radius_ - inner_radius_);

  // Set remaining components to zero
  for (size_t i = 0; i < 3; ++i) {
    jacobian_out->get(i, 2) = 0.0;
  }
  for (size_t i = 0; i < 2; ++i) {
    jacobian_out->get(2, i) = 0.0;
  }
}

template <typename T>
void FlatSide::inv_jacobian(
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

  // Use part of inverse Jacobian for temp storage to avoid allocations.
  const double tmp =
      (2.0 * inner_radius_ - outer_radius_) / (outer_radius_ - inner_radius_);
  // Denominator in the next line is guaranteed to be nonzero.
  get<0, 1>(*inv_jacobian_out) = 1.0 / sqrt(square(xbar) + square(ybar));
  get<0, 1>(*inv_jacobian_out) =
      cube(get<0, 1>(*inv_jacobian_out)) * tmp /
      ((outer_radius_ - inner_radius_) +
       (2.0 * inner_radius_ - outer_radius_) * get<0, 1>(*inv_jacobian_out));

  // dxbar/dx
  get<0, 0>(*inv_jacobian_out) = -square(ybar) * get<0, 1>(*inv_jacobian_out) +
                                 1.0 / (outer_radius_ - inner_radius_);
  // dybar/dy
  get<1, 1>(*inv_jacobian_out) = -square(xbar) * get<0, 1>(*inv_jacobian_out) +
                                 1.0 / (outer_radius_ - inner_radius_);
  // dxbar/dy
  get<0, 1>(*inv_jacobian_out) *= xbar * ybar;
  // dybar/dx
  get<1, 0>(*inv_jacobian_out) = get<0, 1>(*inv_jacobian_out);

  // Set remaining components to zero
  for (size_t i = 0; i < 3; ++i) {
    inv_jacobian_out->get(i, 2) = 0.0;
  }
  for (size_t i = 0; i < 2; ++i) {
    inv_jacobian_out->get(2, i) = 0.0;
  }
}

std::optional<std::array<double, 3>> FlatSide::inverse(
    const std::array<double, 3>& target_coords, const double sigma_in) const {
  const double x = (target_coords[0] - center_[0]);
  const double y = (target_coords[1] - center_[1]);
  const double rho = sqrt(square(x) + square(y));
  const double rhobar =
      (rho - inner_radius_) / (outer_radius_ - inner_radius_) + 1.0;
  if (rhobar < 1.0 and not equal_within_roundoff(rhobar, 1.0)) {
    return {};
  }
  if (rhobar > 2.0 and not equal_within_roundoff(rhobar, 2.0)) {
    return {};
  }
  const double xbar = rhobar * x / rho;
  const double ybar = rhobar * y / rho;
  const double zbar = 2.0 * sigma_in - 1.0;

  if (abs(zbar) > 1.0 and not equal_within_roundoff(abs(zbar), 1.0)) {
    return {};
  }

  return std::array<double, 3>{{xbar, ybar, zbar}};
}

template <typename T>
void FlatSide::sigma(const gsl::not_null<tt::remove_cvref_wrap_t<T>*> sigma_out,
                     const std::array<T, 3>& source_coords) const {
  *sigma_out = 0.5 * (source_coords[2] + 1.0);
}

template <typename T>
void FlatSide::deriv_sigma(
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
void FlatSide::dxbar_dsigma(
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

std::optional<double> FlatSide::lambda_tilde(
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
void FlatSide::deriv_lambda_tilde(
    const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*>
        deriv_lambda_tilde_out,
    const std::array<T, 3>& target_coords, const T& lambda_tilde,
    const std::array<double, 3>& projection_point) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  if constexpr (not std::is_same<ReturnType, double>::value) {
    destructive_resize_components(
        deriv_lambda_tilde_out,
        static_cast<ReturnType>(target_coords[0]).size());
  }
  (*deriv_lambda_tilde_out)[0] = 0.0;
  (*deriv_lambda_tilde_out)[1] = 0.0;
  (*deriv_lambda_tilde_out)[2] =
      -square(lambda_tilde) / (center_[2] - projection_point[2]);
}

void FlatSide::pup(PUP::er& p) {
  p | center_;
  p | inner_radius_;
  p | outer_radius_;
}

bool operator==(const FlatSide& lhs, const FlatSide& rhs) {
  return lhs.center_ == rhs.center_ and
         lhs.inner_radius_ == rhs.inner_radius_ and
         lhs.outer_radius_ == rhs.outer_radius_;
}

bool operator!=(const FlatSide& lhs, const FlatSide& rhs) {
  return not(lhs == rhs);
}
// Explicit instantiations
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template void FlatSide::forward_map(                                        \
      const gsl::not_null<                                                    \
          std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>*>               \
          target_coords,                                                      \
      const std::array<DTYPE(data), 3>& source_coords) const;                 \
  template void FlatSide::jacobian(                                           \
      const gsl::not_null<                                                    \
          tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>*> \
          jacobian_out,                                                       \
      const std::array<DTYPE(data), 3>& source_coords) const;                 \
  template void FlatSide::inv_jacobian(                                       \
      const gsl::not_null<                                                    \
          tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>*> \
          inv_jacobian_out,                                                   \
      const std::array<DTYPE(data), 3>& source_coords) const;                 \
  template void FlatSide::sigma(                                              \
      const gsl::not_null<tt::remove_cvref_wrap_t<DTYPE(data)>*> sigma_out,   \
      const std::array<DTYPE(data), 3>& source_coords) const;                 \
  template void FlatSide::deriv_sigma(                                        \
      const gsl::not_null<                                                    \
          std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>*>               \
          deriv_sigma_out,                                                    \
      const std::array<DTYPE(data), 3>& source_coords) const;                 \
  template void FlatSide::dxbar_dsigma(                                       \
      const gsl::not_null<                                                    \
          std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>*>               \
          dxbar_dsigma_out,                                                   \
      const std::array<DTYPE(data), 3>& source_coords) const;                 \
  template void FlatSide::deriv_lambda_tilde(                                 \
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
