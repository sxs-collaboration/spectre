// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/CleanFunctionsOfTime.hpp"

#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>

#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/Gsl.hpp"

namespace control_system {
void CleanFunctionsOfTimeAction::CleanFunc::apply(
    const gsl::not_null<std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
        functions,
    const double time) {
  for (auto& function : *functions) {
    function.second->truncate_at_time(time);
  }
}

bool CleanFunctionsOfTime::needs_evolved_variables() const { return false; }

PUP::able::PUP_ID CleanFunctionsOfTime::my_PUP_ID = 0;  // NOLINT
}  // namespace control_system
