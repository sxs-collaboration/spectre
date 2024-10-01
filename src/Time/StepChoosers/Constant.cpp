// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/StepChoosers/Constant.hpp"

#include <cmath>
#include <pup.h>
#include <utility>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/TimeStepRequest.hpp"

namespace StepChoosers {
Constant::Constant(const double value) : value_(value) {}

std::pair<TimeStepRequest, bool> Constant::operator()(
    const double last_step) const {
  return {{.size_goal = std::copysign(value_, last_step)}, true};
}

bool Constant::uses_local_data() const { return false; }

bool Constant::can_be_delayed() const { return true; }

void Constant::pup(PUP::er& p) {
  StepChooser<StepChooserUse::Slab>::pup(p);
  StepChooser<StepChooserUse::LtsStep>::pup(p);
  p | value_;
}

PUP::able::PUP_ID Constant::my_PUP_ID = 0;  // NOLINT
}  // namespace StepChoosers

template <>
StepChoosers::Constant
Options::create_from_yaml<StepChoosers::Constant>::create<void>(
    const Options::Option& options) {
  return StepChoosers::Constant{options.parse_as<double>()};
}
