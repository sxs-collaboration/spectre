// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/StepChoosers/Constant.hpp"

#include "Options/ParseOptions.hpp"

namespace StepChoosers {
PUP::able::PUP_ID Constant::my_PUP_ID = 0;  // NOLINT
}  // namespace StepChoosers

template <>
StepChoosers::Constant
Options::create_from_yaml<StepChoosers::Constant>::create<void>(
    const Options::Option& options) {
  return StepChoosers::Constant{options.parse_as<double>()};
}
