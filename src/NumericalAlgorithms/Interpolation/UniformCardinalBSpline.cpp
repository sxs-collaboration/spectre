// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/UniformCardinalBSpline.hpp"

#include <algorithm>
#include <boost/version.hpp>
#include <cmath>
#include <cstddef>
#include <limits>
#include <pup.h>
#include <pup_stl.h>
#include <utility>
#include <vector>

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace intrp {

namespace {

// NOLINTNEXTLINE(clang-diagnostic-missing-noreturn)
void check_boost_version() {
  // Read the version into a volatile so compilers can't prove that this
  // function never returns when compiled with Boost older than 1.81, which
  // would trigger noreturn warnings here and in callers
  const volatile int boost_version = BOOST_VERSION;
  if (boost_version < 108100) {
    ERROR_NO_TRACE(
        "UniformCardinalBSpline requires Boost 1.81 or newer because of a "
        "bugfix in boost::math::interpolators::cardinal_cubic_b_spline, but "
        "found version "
        << BOOST_LIB_VERSION
        << ". See https://github.com/boostorg/math/commit/"
           "4809e714d4806c07da3a3def0c4550daa0529b8d");
  }
}

}  // namespace

UniformCardinalBSpline::UniformCardinalBSpline(std::vector<double> values,
                                               const double start_time,
                                               const double time_step)
    : values_(std::move(values)),
      start_time_(start_time),
      time_step_(time_step) {
  check_boost_version();
  if (values_.size() < 5) {
    ERROR_NO_TRACE(
        "At least 5 samples are required to construct a cubic B-spline, but "
        << values_.size() << " were given.");
  }
  if (not std::isfinite(start_time_)) {
    ERROR_NO_TRACE("The start time must be finite, but is " << start_time_
                                                            << ".");
  }
  if (not std::isfinite(time_step_) or time_step_ <= 0.0) {
    ERROR_NO_TRACE("The time step must be finite and positive, but is "
                   << time_step_ << ".");
  }
  if (const auto non_finite_it = std::find_if(
          values_.begin(), values_.end(),
          [](const double value) { return not std::isfinite(value); });
      non_finite_it != values_.end()) {
    ERROR_NO_TRACE("The sampled values must be finite, but the value at index "
                   << std::distance(values_.begin(), non_finite_it) << " is "
                   << *non_finite_it << ".");
  }
  initialize_interpolant();
}

void UniformCardinalBSpline::initialize_interpolant() {
  if (values_.empty()) {
    // Do nothing for default-initialized objects
    return;
  }
  interpolant_.emplace(values_.begin(), values_.end(), start_time_, time_step_);
}

double UniformCardinalBSpline::operator()(const double time) const {
  ASSERT(interpolant_.has_value(),
         "The interpolant is empty. It was probably default-constructed.");
  const double end_time =
      start_time_ + time_step_ * static_cast<double>(values_.size() - 1);
#ifdef SPECTRE_DEBUG
  const double roundoff_slack =
      100.0 * std::numeric_limits<double>::epsilon() *
      std::max({1.0, std::abs(start_time_), std::abs(end_time)});
  ASSERT(time >= start_time_ - roundoff_slack and
             time <= end_time + roundoff_slack,
         "The time " << time << " to interpolate to is outside the sampled "
                     << "interval [" << start_time_ << ", " << end_time
                     << "].");
#endif
  // Clamp to the sampled interval: the Boost implementation extrapolates the
  // spline outside its knots, but times that fall outside the interval by
  // roundoff should evaluate to the boundary values.
  return interpolant_.value()(std::clamp(time, start_time_, end_time));
}

std::array<double, 2> UniformCardinalBSpline::bounds() const {
  return {{start_time_,
           start_time_ + time_step_ * static_cast<double>(values_.size() - 1)}};
}

void UniformCardinalBSpline::pup(PUP::er& p) {
  p | values_;
  p | start_time_;
  p | time_step_;
  if (p.isUnpacking()) {
    initialize_interpolant();
  }
}

bool operator==(const UniformCardinalBSpline& lhs,
                const UniformCardinalBSpline& rhs) {
  return lhs.values() == rhs.values() and
         lhs.start_time() == rhs.start_time() and
         lhs.time_step() == rhs.time_step();
}

bool operator!=(const UniformCardinalBSpline& lhs,
                const UniformCardinalBSpline& rhs) {
  return not(lhs == rhs);
}

double estimate_interpolation_error(const std::vector<double>& values,
                                    const double start_time,
                                    const double time_step) {
  const size_t num_samples = values.size();
  if (num_samples < 9) {
    ERROR_NO_TRACE(
        "At least 9 samples are required to estimate the interpolation "
        "error, but "
        << num_samples << " were given.");
  }
  const size_t num_coarse_samples = (num_samples + 1) / 2;
  std::vector<double> coarse_values(num_coarse_samples);
  for (size_t i = 0; i < num_coarse_samples; ++i) {
    coarse_values[i] = values[2 * i];
  }
  const UniformCardinalBSpline coarse_interpolant{std::move(coarse_values),
                                                  start_time, 2.0 * time_step};
  // Measure the deviation of the coarse interpolant from the samples at the
  // sample times. An interpolant through all samples would pass through the
  // samples there, so this deviation is also the difference between the two
  // interpolants. For an even number of samples the last sample lies beyond
  // the coarse interpolant's bounds, so skip it.
  const size_t num_compared_samples = 2 * num_coarse_samples - 1;
  double max_deviation = 0.0;
  for (size_t i = 0; i < num_compared_samples; ++i) {
    const double time = start_time + static_cast<double>(i) * time_step;
    max_deviation =
        std::max(max_deviation, std::abs(coarse_interpolant(time) - values[i]));
  }
  // For smooth data the interpolation error of a cubic spline decreases by a
  // factor of 16 when the step size is halved, so the error of an
  // interpolant through all samples is about 1/16 of the coarse
  // interpolant's deviation measured here. Divide by only 8 to obtain a
  // conservative estimate, accounting for nonsmooth data that converges
  // slower.
  return max_deviation / 8.0;
}

std::pair<UniformCardinalBSpline, double> compress_to_tolerance(
    const std::vector<double>& values, const double start_time,
    const double time_step, const double absolute_tolerance) {
  const size_t num_samples = values.size();
  UniformCardinalBSpline reference_interpolant{values, start_time, time_step};
  if (num_samples <= 6) {
    return {std::move(reference_interpolant), 0.0};
  }
  const double time_interval = time_step * static_cast<double>(num_samples - 1);

  const auto resampled_interpolant = [&reference_interpolant, &start_time,
                                      &time_interval](const size_t num_points) {
    const double resampled_time_step =
        time_interval / static_cast<double>(num_points - 1);
    std::vector<double> resampled_values(num_points);
    for (size_t i = 0; i < num_points; ++i) {
      resampled_values[i] = reference_interpolant(
          start_time + static_cast<double>(i) * resampled_time_step);
    }
    return UniformCardinalBSpline{std::move(resampled_values), start_time,
                                  resampled_time_step};
  };
  const auto max_error_at_samples =
      [&values, &start_time, &time_step,
       num_samples](const UniformCardinalBSpline& candidate) {
        double max_error = 0.0;
        for (size_t i = 0; i < num_samples; ++i) {
          const double time = start_time + static_cast<double>(i) * time_step;
          max_error =
              std::max(max_error, std::abs(candidate(time) - values[i]));
        }
        return max_error;
      };

  size_t num_points = 6;
  auto candidate = resampled_interpolant(num_points);
  double max_error = max_error_at_samples(candidate);
  while (max_error > absolute_tolerance and num_points < num_samples) {
    num_points = std::min(2 * num_points, num_samples);
    if (num_points == num_samples) {
      // The tolerance could not be met with fewer points than the input, so
      // return the original samples. Their deviation at the sample times
      // vanishes by construction, so report the error of the last coarser
      // candidate as a conservative estimate.
      return {std::move(reference_interpolant), max_error};
    }
    candidate = resampled_interpolant(num_points);
    max_error = max_error_at_samples(candidate);
  }
  return {std::move(candidate), max_error};
}

}  // namespace intrp
