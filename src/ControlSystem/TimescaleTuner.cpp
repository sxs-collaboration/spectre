// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/TimescaleTuner.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <ostream>

#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

TimescaleTuner::TimescaleTuner(DataVector initial_timescale,
                               const double max_timescale,
                               const double min_timescale,
                               const double decrease_timescale_threshold,
                               const double increase_timescale_threshold,
                               const double increase_factor,
                               const double decrease_factor) noexcept
    : timescale_{std::move(initial_timescale)},
      max_timescale_{max_timescale},
      min_timescale_{min_timescale},
      decrease_timescale_threshold_{decrease_timescale_threshold},
      increase_timescale_threshold_{increase_timescale_threshold},
      increase_factor_{increase_factor},
      decrease_factor_{decrease_factor} {
  for (const auto& t_scale : timescale_) {
    if (t_scale <= 0.0) {
      ERROR("Initial timescale must be > 0");
    }
  }

  if (decrease_factor_ > 1.0 or decrease_factor <= 0.0) {
    ERROR("The specified decrease_factor "
          << decrease_factor_ << " must satisfy 0 < decrease_factor <= 1");
  }
  if (increase_factor_ < 1.0) {
    ERROR("The specified increase factor " << increase_factor_
                                           << " must be >= 1.0");
  }
  if (min_timescale_ <= 0.0) {
    ERROR("The specified minimum timescale " << min_timescale_
                                             << " must be > 0");
  }
  if (max_timescale_ <= min_timescale_) {
    ERROR("The maximum timescale "
          << max_timescale_
          << " must be > than the specified minimum timescale "
          << min_timescale_);
  }
  if (increase_timescale_threshold_ <= 0.0) {
    ERROR("The specified increase-timescale threshold "
          << increase_timescale_threshold_ << " must be > 0");
  }
  if (decrease_timescale_threshold_ <= increase_timescale_threshold_) {
    ERROR("The decrease-timescale threshold "
          << decrease_timescale_threshold_
          << " must be > than the specified increase-timescale threshold "
          << increase_timescale_threshold_);
  }
}

void TimescaleTuner::set_timescale_if_in_allowable_range(
    const double suggested_timescale) noexcept {
  for (auto& t_scale : timescale_) {
    t_scale = cpp17::clamp(suggested_timescale, min_timescale_, max_timescale_);
  }
}

void TimescaleTuner::update_timescale(
    const std::array<DataVector, 2>& q_and_dtq) noexcept {
  ASSERT(q_and_dtq[0].size() == timescale_.size() and
             q_and_dtq[1].size() == timescale_.size(),
         "One or both of the number of components in q_and_dtq("
             << q_and_dtq[0].size() << "," << q_and_dtq[1].size()
             << ") is inconsistent with the number of timescales("
             << timescale_.size() << ")");

  const DataVector& q = gsl::at(q_and_dtq, 0);
  const DataVector& dtq = gsl::at(q_and_dtq, 1);

  for (size_t i = 0; i < q.size(); i++) {
    // check whether we need to decrease the timescale:
    if ((fabs(q[i]) > decrease_timescale_threshold_ or
         fabs(dtq[i] * timescale_[i]) > decrease_timescale_threshold_) and
        (dtq[i] * q[i] > 0.0 or
         fabs(dtq[i]) * timescale_[i] < 0.5 * fabs(q[i]))) {
      // the first check is if Q `or` dtQ are above the maximum tolerance.
      // the second condition of the `and` is
      // that Q and dtQ are the same sign (the error is growing)
      // `or` that Q is not expected to drop to half of its current value in
      // one timescale (not decreasing fast enough)
      timescale_[i] *= decrease_factor_;
    }
    // check whether we need to increase the timescale:
    else if (fabs(q[i]) < increase_timescale_threshold_ and
             fabs(dtq[i] * timescale_[i]) <
                 (increase_timescale_threshold_ - fabs(q[i]))) {
      // if Q `and` dtQ are below the minimum required threshold
      timescale_[i] *= increase_factor_;
    }

    // make sure the timescale has not increased(decreased) above(below) the
    // maximum(minimum) value.
    timescale_[i] = cpp17::clamp(timescale_[i], min_timescale_, max_timescale_);
  }
}
