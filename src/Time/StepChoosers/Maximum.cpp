// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/StepChoosers/Maximum.hpp"

#include <cmath>
#include <pup.h>
#include <utility>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/TimeStepRequest.hpp"

namespace StepChoosers {
Maximum::Maximum(const double value) : value_(value) {}

std::pair<TimeStepRequest, bool> Maximum::operator()(
    const double last_step) const {
  return {{.size = std::copysign(value_, last_step)}, true};
}

bool Maximum::uses_local_data() const { return false; }

bool Maximum::can_be_delayed() const { return true; }

void Maximum::pup(PUP::er& p) {
  StepChooser<StepChooserUse::Slab>::pup(p);
  StepChooser<StepChooserUse::LtsStep>::pup(p);
  p | value_;
}

PUP::able::PUP_ID Maximum::my_PUP_ID = 0;  // NOLINT
}  // namespace StepChoosers

template <>
StepChoosers::Maximum
Options::create_from_yaml<StepChoosers::Maximum>::create<void>(
    const Options::Option& options) {
  return StepChoosers::Maximum{options.parse_as<double>()};
}
