// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/EquatorialCompression.hpp"

#include <cmath>  // IWYU pragma: keep
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain::CoordinateMaps {

EquatorialCompression::EquatorialCompression(const double aspect_ratio,
                                             const size_t index_pole_axis)
    : aspect_ratio_(aspect_ratio),
      inverse_aspect_ratio_(1.0 / aspect_ratio),
      is_identity_(aspect_ratio_ == 1.0),
      index_pole_axis_(index_pole_axis) {
  ASSERT(aspect_ratio > 0.0, "The aspect_ratio must be greater than zero.");
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3>
EquatorialCompression::angular_distortion(const std::array<T, 3>& coords,
                                          const double inverse_alpha) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& x = coords[0];
  const ReturnType& y = coords[1];
  const ReturnType& z = coords[2];
  const ReturnType rho =
      sqrt(square((index_pole_axis_ == 0) ? inverse_alpha * x : x) +
           square((index_pole_axis_ == 1) ? inverse_alpha * y : y) +
           square((index_pole_axis_ == 2) ? inverse_alpha * z : z));

  // While radius_over_rho is set to sqrt(square(x) + square(y) + square(z)),
  // and this is only the radius, the division by rho is handled in the next
  // line.
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
  return std::array<ReturnType, 3>{
      {radius_over_rho * ((index_pole_axis_ == 0) ? inverse_alpha * x : x),
       radius_over_rho * ((index_pole_axis_ == 1) ? inverse_alpha * y : y),
       radius_over_rho * ((index_pole_axis_ == 2) ? inverse_alpha * z : z)}};
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
EquatorialCompression::angular_distortion_jacobian(
    const std::array<T, 3>& coords, const double inverse_alpha) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& x = coords[0];
  const ReturnType& y = coords[1];
  const ReturnType& z = coords[2];
  const ReturnType radius = sqrt(square(x) + square(y) + square(z));
  const ReturnType rho =
      sqrt(square((index_pole_axis_ == 0) ? inverse_alpha * x : x) +
           square((index_pole_axis_ == 1) ? inverse_alpha * y : y) +
           square((index_pole_axis_ == 2) ? inverse_alpha * z : z));

  auto jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(coords[0]), 0.0);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      const size_t k = (index_pole_axis_ + j) % 3;
      jacobian_matrix.get(i, k) = gsl::at(coords, i) * gsl::at(coords, k) *
                                  (square(inverse_alpha) - 1.0);
      if (j == 0) {
        jacobian_matrix.get(i, k) *=
            (square(gsl::at(coords, index_pole_axis_)) - square(radius));
      } else {
        jacobian_matrix.get(i, k) *= square(gsl::at(coords, index_pole_axis_));
      }
      if (i == k) {
        jacobian_matrix.get(i, k) += square(rho * radius);
      }
      if (i == index_pole_axis_) {
        jacobian_matrix.get(i, k) *= inverse_alpha;
      }
    }
  }

  for (size_t i = 0; i < get_size(radius); i++) {
    if (LIKELY(not equal_within_roundoff(get_element(radius, i), 0.0))) {
      const double rho_cubed_i = cube(get_element(rho, i));
      const double radial_factor = 1.0 / (get_element(radius, i) * rho_cubed_i);
      get_element(get<0, 0>(jacobian_matrix), i) *= radial_factor;
      get_element(get<1, 0>(jacobian_matrix), i) *= radial_factor;
      get_element(get<2, 0>(jacobian_matrix), i) *= radial_factor;
      get_element(get<0, 1>(jacobian_matrix), i) *= radial_factor;
      get_element(get<1, 1>(jacobian_matrix), i) *= radial_factor;
      get_element(get<2, 1>(jacobian_matrix), i) *= radial_factor;
      get_element(get<0, 2>(jacobian_matrix), i) *= radial_factor;
      get_element(get<1, 2>(jacobian_matrix), i) *= radial_factor;
      get_element(get<2, 2>(jacobian_matrix), i) *= radial_factor;
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
    const std::array<T, 3>& source_coords) const {
  return angular_distortion(source_coords, inverse_aspect_ratio_);
}

std::optional<std::array<double, 3>> EquatorialCompression::inverse(
    const std::array<double, 3>& target_coords) const {
  return angular_distortion(target_coords, aspect_ratio_);
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
EquatorialCompression::jacobian(const std::array<T, 3>& source_coords) const {
  return angular_distortion_jacobian(source_coords, inverse_aspect_ratio_);
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
EquatorialCompression::inv_jacobian(
    const std::array<T, 3>& source_coords) const {
  return angular_distortion_jacobian((*this)(source_coords), aspect_ratio_);
}

void EquatorialCompression::pup(PUP::er& p) {
  p | aspect_ratio_;
  p | inverse_aspect_ratio_;
  p | index_pole_axis_;
  p | is_identity_;
}

bool operator==(const EquatorialCompression& lhs,
                const EquatorialCompression& rhs) {
  return lhs.aspect_ratio_ == rhs.aspect_ratio_ and
         lhs.inverse_aspect_ratio_ == rhs.inverse_aspect_ratio_ and
         lhs.index_pole_axis_ == rhs.index_pole_axis_ and
         lhs.is_identity_ == rhs.is_identity_;
}

bool operator!=(const EquatorialCompression& lhs,
                const EquatorialCompression& rhs) {
  return not(lhs == rhs);
}

// Explicit instantiations
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>               \
  EquatorialCompression::operator()(                                         \
      const std::array<DTYPE(data), 3>& source_coords) const;                \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  EquatorialCompression::jacobian(                                           \
      const std::array<DTYPE(data), 3>& source_coords) const;                \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  EquatorialCompression::inv_jacobian(                                       \
      const std::array<DTYPE(data), 3>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))
#undef DTYPE
#undef INSTANTIATE
}  // namespace domain::CoordinateMaps
