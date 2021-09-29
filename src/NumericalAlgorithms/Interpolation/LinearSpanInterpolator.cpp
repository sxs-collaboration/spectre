// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/LinearSpanInterpolator.hpp"

#include <complex>

#include "Parallel/CharmPupable.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"

namespace intrp {

namespace {
template <typename ValueType>
SPECTRE_ALWAYS_INLINE ValueType
interpolate_impl(const gsl::span<const double>& source_points,
                 const gsl::span<const ValueType>& values,
                 const double target_point) noexcept {
  return values[0] + (values[1] - values[0]) /
                         (source_points[1] - source_points[0]) *
                         (target_point - source_points[0]);
}
}  // namespace

double LinearSpanInterpolator::interpolate(
    const gsl::span<const double>& source_points,
    const gsl::span<const double>& values, const double target_point) const
    noexcept {
  return interpolate_impl(source_points, values, target_point);
}

std::complex<double> LinearSpanInterpolator::interpolate(
    const gsl::span<const double>& source_points,
    const gsl::span<const std::complex<double>>& values,
    const double target_point) const noexcept {
  return interpolate_impl(source_points, values, target_point);
}

PUP::able::PUP_ID intrp::LinearSpanInterpolator::my_PUP_ID = 0;
}  // namespace intrp
