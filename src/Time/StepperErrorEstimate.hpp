// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>

#include "Time/Time.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// \ingroup TimeGroup
/// Estimate of the TimeStepper truncation error.
struct StepperErrorEstimate {
  /// Start of the step the estimate is for.
  Time step_time{};
  /// Size of the step the estimate is for.
  TimeDelta step_size{};
  /// Order of accuracy of the primary estimate.
  size_t order{};
  /// Error estimates.  The entry `errors[order]`, with `order` the
  /// previous member of this struct, is the estimated error of the
  /// time step, and has a convenience accessor `step_error()`.  Other
  /// entries are estimations of the error if the order of a
  /// variable-order stepper were to be changed.  The error
  /// `errors[i]` should scale approximately as $(\Delta t)^{i + 1}$.
  std::array<std::optional<double>, 8> errors{};

  StepperErrorEstimate() = default;
  StepperErrorEstimate(Time step_time_in, TimeDelta step_size_in,
                       size_t order_in, double step_error_in);

  /// Convenience accessor for `errors[order].value()`.
  double step_error() const;

  void pup(PUP::er& p);
};
