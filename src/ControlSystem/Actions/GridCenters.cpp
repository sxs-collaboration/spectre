// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/Actions/GridCenters.hpp"

#include <array>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/SettleToConstantQuaternion.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/StdHelpers.hpp"

namespace control_system {
void DisableRotationWhen::pup(PUP::er& p) {
  p | disable_at_separation;
  p | rotation_decay_timescale;
}

namespace Tags {
control_system::DisableRotationWhen DisableRotationWhen::create_from_options(
    const control_system::DisableRotationWhen& disable_rotation_when) {
  return disable_rotation_when;
}
}  // namespace Tags

namespace Actions {
void SwitchGridRotationToSettle::UpdateRotationToSettle::apply(
    const gsl::not_null<std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
        f_of_t_list,
    const std::string& function_of_time_name,
    const std::array<DataVector, 3>& initial_func_and_derivs,
    const double match_time, const double decay_time) {
  if (not f_of_t_list->contains(function_of_time_name)) {
    ERROR("Cannot find function of time name '"
          << function_of_time_name
          << "' in the set of functions of time: " << keys_of(*f_of_t_list));
  }
  f_of_t_list->at(function_of_time_name) =
      std::make_unique<domain::FunctionsOfTime::SettleToConstantQuaternion>(
          initial_func_and_derivs, match_time, decay_time);
}

void SwitchGridRotationToSettle::DisableControlSystem::apply(
    const gsl::not_null<std::unordered_map<std::string, bool>*> is_active_map,
    const std::string& control_system_name) {
  if (not is_active_map->contains(control_system_name)) {
    ERROR("Cannot find control system '" << control_system_name
                                         << "' in the active control systems, "
                                         << keys_of(*is_active_map));
  }
  is_active_map->at(control_system_name) = false;
}
}  // namespace Actions
}  // namespace control_system
