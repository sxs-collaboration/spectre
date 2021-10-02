// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/CubicSpline.hpp"

#include <algorithm>
#include <cstddef>
#include <gsl/gsl_spline.h>
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <vector>

#include "Utilities/ErrorHandling/Assert.hpp"

namespace intrp {
CubicSpline::CubicSpline(std::vector<double> x_values,
                         std::vector<double> y_values)
    : x_values_(std::move(x_values)), y_values_(std::move(y_values)) {
  ASSERT(x_values_.size() == y_values_.size(),
         "The x-value and y-value vectors must be of the same length, but "
         "received x-value of size: "
             << x_values_.size()
             << " and y-value of size: " << y_values_.size());
  ASSERT(std::is_sorted(x_values_.begin(), x_values_.end()),
         "The x-values must be sorted.");
  initialize_interpolant();
}

void CubicSpline::gsl_interp_accel_deleter::operator()(
    gsl_interp_accel* const acc) const {
  gsl_interp_accel_free(acc);
}

void CubicSpline::gsl_spline_deleter::operator()(
    gsl_spline* const spline) const {
  gsl_spline_free(spline);
}

void CubicSpline::initialize_interpolant() {
  const size_t num_points = x_values_.size();
  acc_ = std::unique_ptr<gsl_interp_accel, gsl_interp_accel_deleter>{
      gsl_interp_accel_alloc()};
  spline_ = std::unique_ptr<gsl_spline, gsl_spline_deleter>{
      gsl_spline_alloc(gsl_interp_cspline, num_points)};
  gsl_spline_init(spline_.get(), x_values_.data(), y_values_.data(),
                  num_points);
}

double CubicSpline::operator()(const double x_to_interp_to) const {
  ASSERT(
      x_to_interp_to >= x_values_.front() and
          x_to_interp_to <= x_values_.back(),
      "The point "
          << x_to_interp_to
          << " to interpolate to is outside the domain of the interpolation ["
          << x_values_.front() << ", " << x_values_.back() << "].");
  return gsl_spline_eval(spline_.get(), x_to_interp_to, acc_.get());
}

void CubicSpline::pup(PUP::er& p) {
  p | x_values_;
  p | y_values_;
  if (p.isUnpacking()) {
    initialize_interpolant();
  }
}
}  // namespace intrp
