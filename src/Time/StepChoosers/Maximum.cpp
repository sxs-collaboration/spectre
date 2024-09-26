// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/StepChoosers/Maximum.hpp"

#include "Options/ParseOptions.hpp"

namespace StepChoosers {
PUP::able::PUP_ID Maximum::my_PUP_ID = 0;  // NOLINT
}  // namespace StepChoosers

template <>
StepChoosers::Maximum
Options::create_from_yaml<StepChoosers::Maximum>::create<void>(
    const Options::Option& options) {
  return StepChoosers::Maximum{options.parse_as<double>()};
}
