// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "Time/Time.hpp"

/// \ingroup TimeGroup
/// Estimate of the TimeStepper truncation error.
template <typename T>
struct StepperErrorEstimate {
  /// Start of the step the estimate is for.
  Time step_time{};
  /// Size of the step the estimate is for.
  TimeDelta step_size{};
  /// Order of accuracy of the estimate.  The estimated error should
  /// scale approximately as $(\Delta t)^{\text{order} + 1}$.
  size_t order{};
  /// Error estimate, with the same structure as the evolved
  /// variables.
  T error{};

  void pup(PUP::er& p);
};

template <typename T>
void StepperErrorEstimate<T>::pup(PUP::er& p) {
  p | step_time;
  p | step_size;
  p | order;
  p | error;
}
