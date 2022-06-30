// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/Tags/FunctionsOfTimeInitialize.hpp"

#include <memory>
#include <string>
#include <unordered_map>

#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace control_system::Tags::detail {

void check_expiration_time_consistency(
    const std::unordered_map<std::string, double>& initial_expiration_times,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) {
  for (const auto& [name, expr_time] : initial_expiration_times) {
    if (functions_of_time.count(name) == 0) {
      ERROR("The control system '"
            << name
            << "' is not controlling a function of time. Check that the "
               "DomainCreator you have chosen uses all of the control "
               "systems in the executable. The existing functions of time are: "
            << keys_of(functions_of_time));
    }

    if (functions_of_time.at(name)->time_bounds()[1] != expr_time) {
      ERROR("The expiration time for the function of time '"
            << name << "' has been set improperly. It is supposed to be "
            << expr_time << " but is currently set to "
            << functions_of_time.at(name)->time_bounds()[1]
            << ". It is possible that the DomainCreator you are using isn't "
               "compatible with the control systems.");
    }
  }
}
}  // namespace control_system::Tags::detail
