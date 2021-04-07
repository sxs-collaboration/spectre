// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/CubicSpanInterpolator.hpp"
#include "Utilities/ForceInline.hpp"

#include "Parallel/CharmPupable.hpp"

namespace intrp {

namespace {
template <typename ValueType>
SPECTRE_ALWAYS_INLINE ValueType
interpolate_impl(const gsl::span<const double>& source_points,
                 const gsl::span<const ValueType>& values,
                 const double target_point) noexcept {
  const double t0 = source_points[0];
  const double t1 = source_points[1];
  const double t2 = source_points[2];
  const double t3 = source_points[3];

  const auto d0 = values[0];
  const auto d1 = values[1];
  const auto d2 = values[2];
  const auto d3 = values[3];

  return (-((target_point - t2) *
            (d3 * (target_point - t0) * (target_point - t1) * (t0 - t1) *
                 (t0 - t2) * (t1 - t2) +
             (d1 * (target_point - t0) * (t0 - t2) * (t0 - t3) -
              d0 * (target_point - t1) * (t1 - t2) * (t1 - t3)) *
                 (target_point - t3) * (t2 - t3))) +
          d2 * (target_point - t0) * (target_point - t1) * (t0 - t1) *
              (target_point - t3) * (t0 - t3) * (t1 - t3)) /
         ((t0 - t1) * (t0 - t2) * (t1 - t2) * (t0 - t3) * (t1 - t3) *
          (t2 - t3));
}
}  // namespace

double CubicSpanInterpolator::interpolate(
    const gsl::span<const double>& source_points,
    const gsl::span<const double>& values, double target_point) const noexcept {
  return interpolate_impl(source_points, values, target_point);
}

std::complex<double> CubicSpanInterpolator::interpolate(
    const gsl::span<const double>& source_points,
    const gsl::span<const std::complex<double>>& values,
    double target_point) const noexcept {
  return interpolate_impl(source_points, values, target_point);
}

PUP::able::PUP_ID intrp::CubicSpanInterpolator::my_PUP_ID = 0;
}  // namespace intrp
