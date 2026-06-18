// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/Tags/TimeStepper.hpp"

#include <memory>

#include "Time/TimeSteppers/LtsError.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"

namespace Tags {
const LtsTimeStepper& LtsOrError::get(const ::TimeStepper& stepper) {
  if (const auto* const lts_stepper =
          dynamic_cast<const LtsTimeStepper*>(&stepper)) {
    return *lts_stepper;
  } else {
    static const TimeSteppers::LtsError lts_error{};
    return lts_error;
  }
}
}  // namespace Tags
