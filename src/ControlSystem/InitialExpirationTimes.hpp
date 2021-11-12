// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "Utilities/TMPL.hpp"

namespace control_system {
/*!
 * \ingroup ControlSystemGroup
 * \brief Construct the initial expiration times for functions of time that are
 * controlled by a control system
 *
 * The expiration times are constructed using inputs from control system
 * OptionHolders as an unordered map from the name of the function of time being
 * controlled to the expiration time. The expiration time is computed as
 * \f$\tau_\mathrm{exp} = \alpha_\mathrm{update} \tau_\mathrm{damp}\f$ where
 * \f$\alpha_\mathrm{update}\f$ is the update fraction supplied as input to the
 * Controller and \f$\tau_\mathrm{damp}\f$ is/are the damping timescales
 * supplied from the TimescaleTuner (\f$\tau_\mathrm{damp}\f$ is a DataVector
 * with as many components as the corresponding function of time, thus
 * \f$\tau_\mathrm{exp}\f$ will also be a DataVector of the same length).
 *
 */
template <typename... OptionHolders>
std::unordered_map<std::string, double> initial_expiration_times(
    const double initial_time, const double initial_time_step,
    const OptionHolders&... option_holders) {
  std::unordered_map<std::string, double> initial_expiration_times{};

  [[maybe_unused]] const auto gather_initial_expiration_times =
      [&initial_time, &initial_time_step,
       &initial_expiration_times](const auto& option_holder) {
        const auto& controller = option_holder.controller;
        const std::string& name =
            std::decay_t<decltype(option_holder)>::control_system::name();
        const auto& tuner = option_holder.tuner;

        const double update_fraction = controller.get_update_fraction();
        const double curr_timescale = min(tuner.current_timescale());
        const double initial_expiration_time = update_fraction * curr_timescale;
        initial_expiration_times[name] =
            initial_time + std::max(initial_time_step, initial_expiration_time);
      };

  EXPAND_PACK_LEFT_TO_RIGHT(gather_initial_expiration_times(option_holders));

  return initial_expiration_times;
}
}  // namespace control_system
