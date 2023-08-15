// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/FunctionsOfTime/OutputTimeBounds.hpp"

#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Utilities/StdHelpers.hpp"

namespace domain::FunctionsOfTime {
std::string ouput_time_bounds(
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) {
  std::stringstream ss{};
  ss << std::scientific << std::setprecision(16);
  ss << "FunctionsOfTime time bounds:\n";
  for (const auto& [name, function_of_time] : functions_of_time) {
    ss << " " << name << ": " << function_of_time->time_bounds() << "\n";
  }

  return ss.str();
}
}  // namespace domain::FunctionsOfTime
