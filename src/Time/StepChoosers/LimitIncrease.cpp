// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/StepChoosers/LimitIncrease.hpp"

#include <cmath>
#include <pup.h>
#include <utility>

#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/TimeStepRequest.hpp"

namespace StepChoosers {
LimitIncrease::LimitIncrease(const double factor) : factor_(factor) {}

std::pair<TimeStepRequest, bool> LimitIncrease::operator()(
    const double last_step) const {
  return {{.size = last_step * factor_}, true};
}

bool LimitIncrease::uses_local_data() const { return false; }

bool LimitIncrease::can_be_delayed() const { return true; }

void LimitIncrease::pup(PUP::er& p) {
  StepChooser<StepChooserUse::Slab>::pup(p);
  StepChooser<StepChooserUse::LtsStep>::pup(p);
  p | factor_;
}

PUP::able::PUP_ID LimitIncrease::my_PUP_ID = 0;  // NOLINT
}  // namespace StepChoosers
