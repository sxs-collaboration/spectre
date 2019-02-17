// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/EquatorialCompression.hpp"

#include <cmath>  // IWYU pragma: keep
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/SegmentId.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain {
namespace CoordinateMaps {

EquatorialCompression::EquatorialCompression(const double aspect_ratio) noexcept
    : aspect_ratio_(aspect_ratio),
      inverse_aspect_ratio_(1.0 / aspect_ratio),
      is_identity_(aspect_ratio_ == 1.0) {
  ASSERT(aspect_ratio > 0.0, "The aspect_ratio must be greater than zero.");
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3>
EquatorialCompression::angular_distortion(const std::array<T, 3>& coords,
                                          const double inverse_alpha) const
    noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& x = coords[0];
  const ReturnType& y = coords[1];
  const ReturnType& z = coords[2];
  const ReturnType rho =
      sqrt(square(x) + square(y) + square(inverse_alpha * z));
  ReturnType radius_over_rho = sqrt(square(x) + square(y) + square(z));
  for (size_t i = 0; i < get_size(rho); i++) {
    if (LIKELY(get_element(rho, i) != 0.0)) {
      get_element(radius_over_rho, i) /= get_element(rho, i);
    }
    // There is no 'else' covering the case rho==0.  The only way that
    // rho can be zero is if x=y=z=0 (because inverse_alpha is
    // nonzero).  So in that case, what value do we choose for radius_over_rho?
    // Note that radius_over_rho^2 = (x^2+y^2+z^2)/(x^2+y^2+inverse_alpha^2 z^2)
    // does not tend to a limit at the origin. (The limit is
    // 1/inverse_alpha^2 if you approach the origin along the z axis;
    // the limit is 1 if you approach the origin along any path in the
    // xy plane).  But all is ok: notice that radius_over_rho is
    // finite at the origin, and notice that the value returned by this
    // function is multiplied by (x,y,z) below so it will be zero at the
    // origin. Therefore we just leave radius_over_rho unchanged (with
    // a value of zero) in the case rho==0.
  }
  return std::array<ReturnType, 3>{{radius_over_rho * x, radius_over_rho * y,
                                    inverse_alpha * radius_over_rho * z}};
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
EquatorialCompression::angular_distortion_jacobian(
    const std::array<T, 3>& coords, const double inverse_alpha) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& x = coords[0];
  const ReturnType& y = coords[1];
  const ReturnType& z = coords[2];
  const ReturnType radius = sqrt(square(x) + square(y) + square(z));
  const ReturnType rho =
      sqrt(square(x) + square(y) + square(inverse_alpha * z));
  // Various common factors:
  const ReturnType lambda1 = (pow<2>(inverse_alpha) - 1.0) * z * z;
  const ReturnType lambda2 = (1.0 - pow<2>(inverse_alpha)) * (x * x + y * y);
  const ReturnType lambda3 = pow<2>(rho * radius);
  const ReturnType lambda4 = radius * pow<3>(rho);

  auto jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(coords[0]), 0.0);

  get<0, 0>(jacobian_matrix) = (x * x * lambda1 + lambda3);
  get<1, 0>(jacobian_matrix) = x * y * lambda1;
  get<2, 0>(jacobian_matrix) = inverse_alpha * x * z * lambda1;

  get<0, 1>(jacobian_matrix) = get<1, 0>(jacobian_matrix);
  get<1, 1>(jacobian_matrix) = (y * y * lambda1 + lambda3);
  get<2, 1>(jacobian_matrix) = inverse_alpha * y * z * lambda1;

  get<0, 2>(jacobian_matrix) = x * z * lambda2;
  get<1, 2>(jacobian_matrix) = y * z * lambda2;
  get<2, 2>(jacobian_matrix) = inverse_alpha * (z * z * lambda2 + lambda3);

  for (size_t i = 0; i < get_size(radius); i++) {
    if (LIKELY(not equal_within_roundoff(get_element(lambda4, i), 0.0))) {
      const double one_over_lambda4 = 1.0 / get_element(lambda4, i);
      get_element(get<0, 0>(jacobian_matrix), i) *= one_over_lambda4;
      get_element(get<1, 0>(jacobian_matrix), i) *= one_over_lambda4;
      get_element(get<2, 0>(jacobian_matrix), i) *= one_over_lambda4;
      get_element(get<0, 1>(jacobian_matrix), i) *= one_over_lambda4;
      get_element(get<1, 1>(jacobian_matrix), i) *= one_over_lambda4;
      get_element(get<2, 1>(jacobian_matrix), i) *= one_over_lambda4;
      get_element(get<0, 2>(jacobian_matrix), i) *= one_over_lambda4;
      get_element(get<1, 2>(jacobian_matrix), i) *= one_over_lambda4;
      get_element(get<2, 2>(jacobian_matrix), i) *= one_over_lambda4;
    } else {
      // Let the jacobian of this map be the identity at the origin:
      get_element(get<0, 0>(jacobian_matrix), i) = 1.0;
      get_element(get<1, 0>(jacobian_matrix), i) = 0.0;
      get_element(get<2, 0>(jacobian_matrix), i) = 0.0;
      get_element(get<0, 1>(jacobian_matrix), i) = 0.0;
      get_element(get<1, 1>(jacobian_matrix), i) = 1.0;
      get_element(get<2, 1>(jacobian_matrix), i) = 0.0;
      get_element(get<0, 2>(jacobian_matrix), i) = 0.0;
      get_element(get<1, 2>(jacobian_matrix), i) = 0.0;
      get_element(get<2, 2>(jacobian_matrix), i) = 1.0;
    }
  }
  return jacobian_matrix;
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> EquatorialCompression::operator()(
    const std::array<T, 3>& source_coords) const noexcept {
  return angular_distortion(source_coords, inverse_aspect_ratio_);
}

boost::optional<std::array<double, 3>> EquatorialCompression::inverse(
    const std::array<double, 3>& target_coords) const noexcept {
  return angular_distortion(target_coords, aspect_ratio_);
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
EquatorialCompression::jacobian(const std::array<T, 3>& source_coords) const
    noexcept {
  return angular_distortion_jacobian(source_coords, inverse_aspect_ratio_);
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
EquatorialCompression::inv_jacobian(const std::array<T, 3>& source_coords) const
    noexcept {
  return angular_distortion_jacobian((*this)(source_coords), aspect_ratio_);
}

void EquatorialCompression::pup(PUP::er& p) noexcept {
  p | aspect_ratio_;
  p | inverse_aspect_ratio_;
  p | is_identity_;
}

bool operator==(const EquatorialCompression& lhs,
                const EquatorialCompression& rhs) noexcept {
  return lhs.aspect_ratio_ == rhs.aspect_ratio_ and
         lhs.inverse_aspect_ratio_ == rhs.inverse_aspect_ratio_ and
         lhs.is_identity_ == rhs.is_identity_;
}

bool operator!=(const EquatorialCompression& lhs,
                const EquatorialCompression& rhs) noexcept {
  return not(lhs == rhs);
}

// Explicit instantiations
/// \cond
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>               \
  EquatorialCompression::operator()(                                         \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;       \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  EquatorialCompression::jacobian(                                           \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;       \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  EquatorialCompression::inv_jacobian(                                       \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))
#undef DTYPE
#undef INSTANTIATE
/// \endcond
}  // namespace CoordinateMaps
}  // namespace domain
