// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Burgers/Bump.hpp"

#include <cmath>
#include <pup.h>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace Burgers {
namespace Solutions {

Bump::Bump(const double half_width, const double height,
           const double center) noexcept
    : half_width_(half_width), height_(height), center_(center) {}

template <typename T>
Scalar<T> Bump::u(const tnsr::I<T, 1>& x, double t) const noexcept {
  const T center_distance = get<0>(x) - center_;
  // Distance from the current peak location divided by the half width
  // and the time the shock reaches the solution zero.
  const T reduced_peak_distance =
      2. * height_ / square(half_width_) * (center_distance - height_ * t);

  const T denom = 1. / (1. + sqrt(1. - 2. * t * reduced_peak_distance));

  return Scalar<T>(2. * denom *
                   (height_ - center_distance * reduced_peak_distance * denom));
}

template <typename T>
Scalar<T> Bump::du_dt(const tnsr::I<T, 1>& x, double t) const noexcept {
  const T center_distance = get<0>(x) - center_;
  // Distance from the current peak location divided by the half width
  // and the time the shock reaches the solution zero.
  const T reduced_peak_distance =
      2. * height_ / square(half_width_) * (center_distance - height_ * t);

  const T denom = 1. / (1. + sqrt(1. - 2. * t * reduced_peak_distance));

  return Scalar<T>(4. * square(height_ / half_width_) * square(denom) *
                   (denom / (1. - denom) *
                        (center_distance - 2. * height_ * t) *
                        (1 - 2. / height_ * denom * center_distance *
                                 reduced_peak_distance) +
                    center_distance));
}

tuples::TaggedTuple<Tags::U> Bump::variables(
    const tnsr::I<DataVector, 1>& x, double t,
    tmpl::list<Tags::U> /*meta*/) const noexcept {
  return {u(x, t)};
}

tuples::TaggedTuple<::Tags::dt<Tags::U>> Bump::variables(
    const tnsr::I<DataVector, 1>& x, double t,
    tmpl::list<::Tags::dt<Tags::U>> /*meta*/) const noexcept {
  return {du_dt(x, t)};
}

void Bump::pup(PUP::er& p) noexcept {
  p | half_width_;
  p | height_;
  p | center_;
}

}  // namespace Solutions
}  // namespace Burgers

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                      \
  template Scalar<DTYPE(data)> Burgers::Solutions::Bump::u(       \
      const tnsr::I<DTYPE(data), 1>& x, double t) const noexcept; \
  template Scalar<DTYPE(data)> Burgers::Solutions::Bump::du_dt(   \
      const tnsr::I<DTYPE(data), 1>& x, double t) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
