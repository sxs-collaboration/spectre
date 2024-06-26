// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

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
  /// Order of accuracy of the estimate.  The estimated error should
  /// scale approximately as $(\Delta t)^{\text{order} + 1}$.
  size_t order{};
  /// Error estimate.
  double error{};

  void pup(PUP::er& p);
};
