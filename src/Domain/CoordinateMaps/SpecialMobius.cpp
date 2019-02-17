// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/SpecialMobius.hpp"

#include <boost/none.hpp>
#include <boost/optional.hpp>
#include <cmath>
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/SegmentId.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace domain {
namespace CoordinateMaps {

SpecialMobius::SpecialMobius(const double mu) noexcept
    : mu_(mu), is_identity_(mu_ == 0.0) {
  // Note: Empirically we have found that the map is accurate
  // to 12 decimal places for mu = 0.96.
  ASSERT(abs(mu) < 0.96, "The magnitude of mu must be less than 0.96.");
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> SpecialMobius::mobius_distortion(
    const std::array<T, 3>& coords, const double mu) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& x = coords[0];
  const ReturnType& y = coords[1];
  const ReturnType& z = coords[2];
  const double mu_squared = square(mu);
  const ReturnType r_squared = square(x) + square(y) + square(z);
  const ReturnType lambda = 1.0 / (1.0 - 2.0 * mu * x + mu_squared * r_squared);
  return std::array<ReturnType, 3>{
      {lambda * ((1.0 + mu_squared) * x - mu * (1.0 + r_squared)),
       (1.0 - mu_squared) * lambda * y, (1.0 - mu_squared) * lambda * z}};
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
SpecialMobius::mobius_distortion_jacobian(const std::array<T, 3>& coords,
                                          const double mu) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& x = coords[0];
  const ReturnType& y = coords[1];
  const ReturnType& z = coords[2];
  const double mu_squared = square(mu);
  const ReturnType r_squared = square(x) + square(y) + square(z);
  const ReturnType common_factor =
      (mu_squared - 1.0) / square(1.0 - 2.0 * mu * x + mu_squared * r_squared);
  auto jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(coords[0]), 0.0);

  get<0, 0>(jacobian_matrix) =
      -(mu_squared * (2.0 * square(x) - r_squared) - 2.0 * mu * x + 1.0) *
      common_factor;
  get<1, 0>(jacobian_matrix) = (2.0 * mu * y * (mu * x - 1.0)) * common_factor;
  get<2, 0>(jacobian_matrix) = (2.0 * mu * z * (mu * x - 1.0)) * common_factor;

  get<0, 1>(jacobian_matrix) = -1.0 * get<1, 0>(jacobian_matrix);
  get<1, 1>(jacobian_matrix) =
      -(mu_squared * (r_squared - 2.0 * square(y)) - 2.0 * mu * x + 1.0) *
      common_factor;
  get<2, 1>(jacobian_matrix) = 2.0 * mu_squared * y * z * common_factor;

  get<0, 2>(jacobian_matrix) = -1.0 * get<2, 0>(jacobian_matrix);
  get<1, 2>(jacobian_matrix) = get<2, 1>(jacobian_matrix);
  get<2, 2>(jacobian_matrix) =
      -(mu_squared * (r_squared - 2.0 * square(z)) - 2.0 * mu * x + 1.0) *
      common_factor;
  return jacobian_matrix;
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> SpecialMobius::operator()(
    const std::array<T, 3>& source_coords) const noexcept {
  return mobius_distortion(source_coords, mu_);
}

boost::optional<std::array<double, 3>> SpecialMobius::inverse(
    const std::array<double, 3>& target_coords) const noexcept {
  // Invert only points inside or on the unit sphere.
  const auto r_squared = magnitude(target_coords);
  if (r_squared <= 1.0 or equal_within_roundoff(r_squared,1.0)) {
    return mobius_distortion(target_coords, -mu_);
  }
  return boost::none;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> SpecialMobius::jacobian(
    const std::array<T, 3>& source_coords) const noexcept {
  return mobius_distortion_jacobian(source_coords, mu_);
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
SpecialMobius::inv_jacobian(const std::array<T, 3>& source_coords) const
    noexcept {
  return mobius_distortion_jacobian((*this)(source_coords), -mu_);
}

void SpecialMobius::pup(PUP::er& p) noexcept {
  p | mu_;
  p | is_identity_;
}

bool operator==(const SpecialMobius& lhs, const SpecialMobius& rhs) noexcept {
  return lhs.mu_ == rhs.mu_ and lhs.is_identity_ == rhs.is_identity_;
}

bool operator!=(const SpecialMobius& lhs, const SpecialMobius& rhs) noexcept {
  return not(lhs == rhs);
}

// Explicit instantiations
/// \cond
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3> SpecialMobius:: \
  operator()(const std::array<DTYPE(data), 3>& source_coords) const noexcept;  \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>   \
  SpecialMobius::jacobian(const std::array<DTYPE(data), 3>& source_coords)     \
      const noexcept;                                                          \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>   \
  SpecialMobius::inv_jacobian(const std::array<DTYPE(data), 3>& source_coords) \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))

#undef DTYPE
#undef INSTANTIATE
/// \endcond
}  // namespace CoordinateMaps
}  // namespace domain
