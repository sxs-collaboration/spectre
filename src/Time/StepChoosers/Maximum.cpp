// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/StepChoosers/Maximum.hpp"

#include "Options/ParseOptions.hpp"

namespace StepChoosers::Maximum_detail {
// This function lets us avoid including ParseOptions.hpp in the
// header.
double parse_options(const Options::Option& options) {
  return options.parse_as<double>();
}
}  // namespace StepChoosers::Maximum_detail
