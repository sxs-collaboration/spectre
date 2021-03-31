// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/FocallyLiftedMap.hpp"

#include <cmath>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedEndcap.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedFlatEndcap.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedFlatSide.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedMapHelpers.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedSide.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain::CoordinateMaps {

template <typename InnerMap>
FocallyLiftedMap<InnerMap>::FocallyLiftedMap(
    const std::array<double, 3>& center,
    const std::array<double, 3>& proj_center, double radius,
    const bool source_is_between_focus_and_target, InnerMap inner_map) noexcept
    : center_(center),
      proj_center_(proj_center),
      radius_(radius),
      source_is_between_focus_and_target_(source_is_between_focus_and_target),
      inner_map_(std::move(inner_map)) {}

template <typename InnerMap>
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> FocallyLiftedMap<InnerMap>::
operator()(const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  // Temporary variable that will be re-used to avoid extra
  // allocations.
  ReturnType temp_scalar{};

  // lower_coords are the mapped coords on the surface.
  std::array<ReturnType, 3> lower_coords{};
  inner_map_.forward_map(&lower_coords, source_coords);

  // upper_coords are the mapped coords on the surface of the sphere.
  ReturnType& lambda = temp_scalar;
  FocallyLiftedMapHelpers::scale_factor(
      &lambda, lower_coords, proj_center_, center_, radius_,
      source_is_between_focus_and_target_);

  std::array<ReturnType, 3> upper_coords{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(upper_coords, i) =
        gsl::at(proj_center_, i) +
        (gsl::at(lower_coords, i) - gsl::at(proj_center_, i)) * lambda;
  }

  // mapped_coords goes linearly from lower_coords to upper_coords
  // as sigma goes from 0 to 1.
  ReturnType& sigma = temp_scalar; // sigma shares memory with lambda.
  inner_map_.sigma(&sigma, source_coords);

  // Use upper_coords to store result, so as to save an allocation.
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(upper_coords, i) =
        gsl::at(lower_coords, i) +
        (gsl::at(upper_coords, i) - gsl::at(lower_coords, i)) * sigma;
  }
  return upper_coords;
}

template <typename InnerMap>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
FocallyLiftedMap<InnerMap>::jacobian(
    const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  // Use these variables to reduce allocations.
  std::array<ReturnType, 3> temp_vector_one{};
  std::array<ReturnType, 3> temp_vector_two{};

  auto jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);

  // lower_coords are the mapped coords on the surface.
  std::array<ReturnType, 3> lower_coords{};
  inner_map_.forward_map(&lower_coords, source_coords);

  ReturnType lambda{};
  FocallyLiftedMapHelpers::scale_factor(
      &lambda, lower_coords, proj_center_, center_, radius_,
      source_is_between_focus_and_target_);

  // upper_coords are the mapped coords on the surface of the sphere.
  std::array<ReturnType, 3>& upper_coords = temp_vector_one;
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(upper_coords, i) =
        gsl::at(proj_center_, i) +
        (gsl::at(lower_coords, i) - gsl::at(proj_center_, i)) * lambda;
  }

  std::array<ReturnType, 3>& d_lambda_d_lower_coords = temp_vector_two;
  FocallyLiftedMapHelpers::d_scale_factor_d_src_point<ReturnType>(
      &d_lambda_d_lower_coords, upper_coords, proj_center_, center_, lambda);

  // Put the inner map's jacobian temporarily into jacobian_matrix.
  // This saves an allocation.
  inner_map_.jacobian(&jacobian_matrix, source_coords);

  // Re-use memory of upper_coords in computing sigma_d_lambda_d_xbar.
  // Don't multiply by sigma yet because we don't know it.
  std::array<ReturnType, 3>& sigma_d_lambda_d_xbar = temp_vector_one;
  for (size_t j = 0; j < 3; ++j) {
    gsl::at(sigma_d_lambda_d_xbar, j) =
        d_lambda_d_lower_coords[0] * jacobian_matrix.get(0, j);
    for (size_t k = 1; k < 3; ++k) {  // First iteration split out above.
      gsl::at(sigma_d_lambda_d_xbar, j) +=
          gsl::at(d_lambda_d_lower_coords, k) * jacobian_matrix.get(k, j);
    }
  }

  // temp_vector_two isn't used anymore, so use its memory for sigma.
  ReturnType& sigma = temp_vector_two[0];
  inner_map_.sigma(&sigma, source_coords);

  // Now complete computation of sigma_d_lambda_d_xbar.
  for (size_t j = 0; j < 3; ++j) {
    gsl::at(sigma_d_lambda_d_xbar, j) *= sigma;
  }

  // Do the easiest of the terms involving the inner map,
  // i.e. the first term in Eq. (6) in the documentation.
  ReturnType& lambda_factor = temp_vector_two[1];
  lambda_factor = 1.0 - sigma + lambda * sigma;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      jacobian_matrix.get(i, j) *= lambda_factor;
    }
  }

  // Do the deriv sigma term, the third term in Eq. (6) in the
  // documentation.
  // Note that we explicitly substitute for upper_coords below
  // because we have re-used its memory.
  std::array<ReturnType, 3>& d_sigma = temp_vector_two;
  inner_map_.deriv_sigma(&d_sigma, source_coords);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      jacobian_matrix.get(i, j) +=
          gsl::at(d_sigma, j) *
          (gsl::at(proj_center_, i) +
           (gsl::at(lower_coords, i) - gsl::at(proj_center_, i)) * lambda -
           gsl::at(lower_coords, i));
    }
  }

  // Do the second term in Eq. (6) in the documentation.
  for (size_t j = 0; j < 3; ++j) {
    for (size_t i = 0; i < 3; ++i) {
      jacobian_matrix.get(i, j) +=
          gsl::at(sigma_d_lambda_d_xbar, j) *
          (gsl::at(lower_coords, i) - gsl::at(proj_center_, i));
    }
  }

  return jacobian_matrix;
}

template <typename InnerMap>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
FocallyLiftedMap<InnerMap>::inv_jacobian(
    const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  // lower_coords are the mapped coords on the surface.
  std::array<ReturnType, 3> lower_coords{};
  inner_map_.forward_map(&lower_coords, source_coords);

  ReturnType lambda{};
  FocallyLiftedMapHelpers::scale_factor(
      &lambda, lower_coords, proj_center_, center_, radius_,
      source_is_between_focus_and_target_);

  // upper_coords are the mapped coords on the surface of the sphere.
  std::array<ReturnType, 3> upper_coords{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(upper_coords, i) =
        gsl::at(proj_center_, i) +
        (gsl::at(lower_coords, i) - gsl::at(proj_center_, i)) * lambda;
  }

  // Derivative of lambda
  std::array<ReturnType, 3> d_lambda_d_lower_coords{};
  FocallyLiftedMapHelpers::d_scale_factor_d_src_point<ReturnType>(
      &d_lambda_d_lower_coords, upper_coords, proj_center_, center_, lambda);

  // Lambda_tilde is the scale factor between mapped coords and lower coords.
  // We can compute it with a shortcut because there is a relationship
  // between lambda, lambda_tilde, and sigma.
  ReturnType sigma {};
  inner_map_.sigma(&sigma, source_coords);
  const ReturnType lambda_tilde = 1.0 / (1.0 - sigma * (1.0 - lambda));

  // Derivative of lambda_tilde
  std::array<ReturnType, 3> d_lambda_tilde_d_mapped_coords{};
  inner_map_.deriv_lambda_tilde(&d_lambda_tilde_d_mapped_coords, lower_coords,
                                lambda_tilde, proj_center_);

  // Deriv of x_0 with respect to x
  auto dx_inner_dx =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      dx_inner_dx.get(i, j) = gsl::at(d_lambda_tilde_d_mapped_coords, j) *
                         (gsl::at(lower_coords, i) - gsl::at(proj_center_, i)) /
                         lambda_tilde;
    }
    dx_inner_dx.get(i, i) += lambda_tilde;
  }

  // Deriv of sigma with respect to x,y,z
  auto d_sigma_d_mapped_coords =
      make_with_value<tnsr::i<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          sigma, 0.0);
  ReturnType tmp{};
  for (size_t i = 0; i < 3; ++i) {
    tmp = d_lambda_d_lower_coords[0] * dx_inner_dx.get(0, i);
    for (size_t j = 1; j < 3; ++j) {  // first iteration factored out above.
      tmp += gsl::at(d_lambda_d_lower_coords, j) * dx_inner_dx.get(j, i);
    }
    d_sigma_d_mapped_coords.get(i) =
        (sigma * tmp +
         gsl::at(d_lambda_tilde_d_mapped_coords, i) / square(lambda_tilde)) /
        (1.0 - lambda);
  }

  tnsr::Ij<ReturnType, 3, Frame::NoFrame> dxbar_dx_inner{};
  inner_map_.inv_jacobian(&dxbar_dx_inner, source_coords);
  std::array<tt::remove_cvref_wrap_t<T>, 3> dxbar_dsigma{};
  inner_map_.dxbar_dsigma(&dxbar_dsigma, source_coords);

  auto inv_jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        inv_jacobian_matrix.get(i, j) +=
            dxbar_dx_inner.get(i, k) * dx_inner_dx.get(k, j);
      }
      inv_jacobian_matrix.get(i, j) +=
          gsl::at(dxbar_dsigma, i) * d_sigma_d_mapped_coords.get(j);
    }
  }

  return inv_jacobian_matrix;
}

template <typename InnerMap>
std::optional<std::array<double, 3>> FocallyLiftedMap<InnerMap>::inverse(
    const std::array<double, 3>& target_coords) const noexcept {
  // Scale factor taking target_coords to lower_coords.
  const auto lambda_tilde = inner_map_.lambda_tilde(
      target_coords, proj_center_, source_is_between_focus_and_target_);

  // Cannot find scale factor, so we are out of range of the map.
  if (not lambda_tilde) {
    return std::optional<std::array<double, 3>>{};
  }

  // Try to find lambda_bar going from target_coords to sphere.
  const auto lambda_bar = FocallyLiftedMapHelpers::try_scale_factor(
      target_coords, proj_center_, center_, radius_,
      not source_is_between_focus_and_target_,
      source_is_between_focus_and_target_);

  // Cannot find scale factor, so we are out of range of the map.
  if (not lambda_bar) {
    return std::optional<std::array<double, 3>>{};
  }

  // compute sigma in a roundoff-friendly way.
  double sigma = 0.0;
  if (equal_within_roundoff(*lambda_tilde, 1.0, 1.e-5)) {
    // Get sigma correct for sigma near 0
    sigma =
        (*lambda_tilde - 1.0) / (*lambda_tilde - *lambda_bar);
  } else {
    // Get sigma correct for sigma near 1
    sigma = (*lambda_bar - 1.0) / (*lambda_tilde - *lambda_bar) +
            1.0;
  }

  std::array<double, 3> lower_coords{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(lower_coords, i) =
        gsl::at(proj_center_, i) +
        (gsl::at(target_coords, i) - gsl::at(proj_center_, i)) *
            lambda_tilde.value();
  }

  std::optional<std::array<double, 3>> orig_coords =
      inner_map_.inverse(lower_coords, sigma);

  // Root polishing.
  // Here we do a single Newton iteration to get the
  // inverse to agree with the forward map to the level of machine
  // roundoff that is required by the unit tests.
  // Without the root polishing, the unit tests occasionally fail
  // the 'inverse(map(x))=x' test at a level slightly above roundoff.
  if (orig_coords) {
    const auto inv_jac = inv_jacobian(*orig_coords);
    const auto mapped_coords = operator()(*orig_coords);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        gsl::at(*orig_coords, i) +=
            (gsl::at(target_coords, j) - gsl::at(mapped_coords, j)) *
            inv_jac.get(i, j);
      }
    }
  }

  return orig_coords;
}

template <typename InnerMap>
void FocallyLiftedMap<InnerMap>::pup(PUP::er& p) noexcept {
  p | center_;
  p | proj_center_;
  p | radius_;
  p | source_is_between_focus_and_target_;
  p | inner_map_;
}

template <typename InnerMap>
bool operator!=(const FocallyLiftedMap<InnerMap>& lhs,
                const FocallyLiftedMap<InnerMap>& rhs) noexcept {
  return not(lhs == rhs);
}

template <typename InnerMap>
bool operator==(const FocallyLiftedMap<InnerMap>& lhs,
                const FocallyLiftedMap<InnerMap>& rhs) noexcept {
  return lhs.center_ == rhs.center_ and lhs.proj_center_ == rhs.proj_center_ and
         lhs.radius_ == rhs.radius_ and
         lhs.source_is_between_focus_and_target_ ==
             rhs.source_is_between_focus_and_target_ and
         lhs.inner_map_ == rhs.inner_map_;
}

// Explicit instantiations
/// \cond
#define IMAP(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template class FocallyLiftedMap<IMAP(data)>;                                \
  template bool operator==(const FocallyLiftedMap<IMAP(data)>& lhs,           \
                           const FocallyLiftedMap<IMAP(data)>& rhs) noexcept; \
  template bool operator!=(const FocallyLiftedMap<IMAP(data)>& lhs,           \
                           const FocallyLiftedMap<IMAP(data)>& rhs) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (FocallyLiftedInnerMaps::Endcap,
                                      FocallyLiftedInnerMaps::FlatEndcap,
                                      FocallyLiftedInnerMaps::FlatSide,
                                      FocallyLiftedInnerMaps::Side))

#undef INSTANTIATE

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                 \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>               \
  FocallyLiftedMap<IMAP(data)>::operator()(                                  \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;       \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  FocallyLiftedMap<IMAP(data)>::jacobian(                                    \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;       \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  FocallyLiftedMap<IMAP(data)>::inv_jacobian(                                \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (FocallyLiftedInnerMaps::Endcap,
                         FocallyLiftedInnerMaps::FlatEndcap,
                         FocallyLiftedInnerMaps::FlatSide,
                         FocallyLiftedInnerMaps::Side),
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))

#undef INSTANTIATE
#undef DTYPE
#undef IMAP
/// \endcond

}  // namespace domain::CoordinateMaps
