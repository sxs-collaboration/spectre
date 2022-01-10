// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"

#include <boost/math/interpolators/barycentric_rational.hpp>
#include <complex>
#include <cstddef>

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

// Doesn't exist in boost before 1.77, but referenced below.
namespace boost::math::interpolators {}

namespace intrp {

BarycentricRationalSpanInterpolator::BarycentricRationalSpanInterpolator(
    size_t min_order, size_t max_order)
    : min_order_{min_order}, max_order_{max_order} {
  ASSERT(min_order <= max_order,
         "The minimum order for the Barycentric rational interpolator must be "
         "less than the maximum order.");
}

double BarycentricRationalSpanInterpolator::interpolate(
    const gsl::span<const double>& source_points,
    const gsl::span<const double>& values, const double target_point) const {
  if (UNLIKELY(source_points.size() < min_order_ + 1)) {
    ERROR("provided independent values for interpolation too small.");
  }
  // Boost moved barycentric_rational from boost::math to
  // boost::math::interpolators in version 1.77.
  // NOLINTNEXTLINE(google-build-using-namespace)
  using namespace boost::math;
  // NOLINTNEXTLINE(google-build-using-namespace)
  using namespace boost::math::interpolators;
  barycentric_rational<double> interpolant(
      source_points.data(), values.data(), source_points.size(),
      std::min(source_points.size() - 1, max_order_));
  return interpolant(target_point);
}

PUP::able::PUP_ID intrp::BarycentricRationalSpanInterpolator::my_PUP_ID = 0;
}  // namespace intrp
