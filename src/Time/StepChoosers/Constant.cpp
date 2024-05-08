// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/StepChoosers/Constant.hpp"

#include "Options/ParseOptions.hpp"

namespace StepChoosers {
namespace Constant_detail {
// This function lets us avoid including ParseOptions.hpp in the
// header.
double parse_options(const Options::Option& options) {
  return options.parse_as<double>();
}
}  // namespace Constant_detail
}  // namespace StepChoosers
