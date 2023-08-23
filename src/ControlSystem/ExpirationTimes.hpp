// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include "ControlSystem/CalculateMeasurementTimescales.hpp"
#include "ControlSystem/CombinedName.hpp"
#include "ControlSystem/Metafunctions.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Utilities/TMPL.hpp"

namespace control_system {
/*!
 * \ingroup ControlSystemGroup
 * \brief Calculate the next expiration time for the FunctionsOfTime.
 *
 * \f{align}
 * T_\mathrm{expr}^\mathrm{FoT} &= t + \tau_\mathrm{m}^\mathrm{old}
 *      + N * \tau_\mathrm{m}^\mathrm{new} \\
 * \f}
 *
 * where \f$T_\mathrm{expr}^\mathrm{FoT}\f$ is the expiration time for the
 * FunctionsOfTime, \f$t\f$ is the current time,
 * \f$\tau_\mathrm{m}^\mathrm{old/new}\f$ is the measurement timescale, and
 * \f$N\f$ is the number of measurements per update.
 *
 * The expiration is calculated this way because we update the functions of time
 * one (old) measurement before they actually expire.
 *
 * The choice of having the functions of time expire exactly one old measurement
 * after they are updated is arbitrary. They could expire any time between the
 * update time and one old measurement after the update. This decision was made
 * to minimize time spent waiting for the functions of time to be valid.
 *
 * Since functions of time are valid at their expiration time, we are actually
 * able to do the next measurement if the expiration time is at that
 * measurement. Thus we delay any potential waiting that may happen until the
 * subsequent measurement (and by that time, most, if not all, functions of time
 * should have been updated because an entire horizon find happened in the
 * meantime). If the expiration time was earlier than the next measurement, we'd
 * have to pause the Algorithm on the DG elements and wait until all the
 * functions of time have been updated.
 */
double function_of_time_expiration_time(
    const double time, const DataVector& old_measurement_timescales,
    const DataVector& new_measurement_timescales,
    const int measurements_per_update);

/*!
 * \ingroup ControlSystemGroup
 * \brief Calculate the next expiration time for the MeasurementTimescales.
 *
 * \f{align}
 * T_\mathrm{expr}^\mathrm{m} &= T_\mathrm{expr}^\mathrm{FoT} - \frac{1}{2}
 *   \tau_\mathrm{m}^\mathrm{new} \\
 * \f}
 *
 * where \f$T_\mathrm{expr}^\mathrm{m}\f$ is the measurement expiration time,
 * \f$T_\mathrm{expr}^\mathrm{FoT}\f$ is the function of time expiration time as
 * calculate from `function_of_time_expiration_time()`, and
 * \f$\tau_\mathrm{m}^\mathrm{new}\f$ is the new measurement timescale. The
 * reason for the factor of a half is as follows:
 *
 * We update the functions of time one (old) measurement before the expiration
 * time. Based on how dense triggers are set up, which control_system::Trigger
 * is a dense trigger, you calculate the next trigger (measurement) time at the
 * current measurement time. However, at the function of time expiration time we
 * need updated damping timescales from all control systems in order to
 * calculate when the next measurement is going to be (and in turn, the next
 * measurement expiration time). Thus, at the measurement that occurs at the
 * function of time expiration time, our measurement timescales can't be valid
 * and we must wait for updated ones. To achieve this, we set the measurement
 * expiration time *before* the function of time expiration time, but *after*
 * the previous measurement (the update measurement). The factor of one half is
 * just to guarantee we are more than epsilon before the function of time
 * expiration time and more than epsilon after the update measurement.
 */
double measurement_expiration_time(const double time,
                                   const DataVector& old_measurement_timescales,
                                   const DataVector& new_measurement_timescales,
                                   const int measurements_per_update);

/*!
 * \ingroup ControlSystemGroup
 * \brief Construct the initial expiration times for functions of time that are
 * controlled by a control system
 *
 * The expiration times are constructed using inputs from control system
 * OptionHolders as an unordered map from the name of the function of time being
 * controlled to the expiration time.
 *
 * The expiration time for each individual function of time is computed as
 * $\tau_\mathrm{exp} = \alpha_\mathrm{update} \tau_\mathrm{damp}$ where
 * $\alpha_\mathrm{update}$ is the update fraction supplied as input to the
 * Controller and $\tau_\mathrm{damp}$ is/are the damping timescales
 * supplied from the TimescaleTuner ($\tau_\mathrm{damp}$ is a DataVector
 * with as many components as the corresponding function of time, thus
 * $\tau_\mathrm{exp}$ will also be a DataVector of the same length).
 *
 * However, this expiration time calculated above is not necessarily the
 * expiration that is returned by this function. We group functions of time by
 * the `control_system::protocols::Measurement` that their corresponding control
 * systems use. This is because one measurement may be used to update many
 * functions of time. So the actual expiration time that is used for all the
 * functions of time in this group is the *minimum* of the $\tau_\mathrm{exp}$
 * of each function of time in the group.
 *
 * These groups are calculated using `control_system::system_to_combined_names`
 * where the list of control systems comes from the passed in `OptionHolder`s.
 *
 * If the control system isn't active then expiration time is
 * `std::numeric_limits<double>::infinity()`, regardless of what the groups'
 * expiration time is.
 */
template <size_t Dim, typename... OptionHolders>
std::unordered_map<std::string, double> initial_expiration_times(
    const double initial_time, const int measurements_per_update,
    const std::unique_ptr<::DomainCreator<Dim>>& domain_creator,
    const OptionHolders&... option_holders) {
  std::unordered_map<std::string, double> initial_expiration_times{};

  using control_systems = tmpl::list<typename OptionHolders::control_system...>;

  // First string is name of each control system. Second string is combination
  // of control systems with same measurement
  std::unordered_map<std::string, std::string> map_of_names =
      system_to_combined_names<control_systems>();

  std::unordered_map<std::string, double> combined_expiration_times{};
  std::unordered_set<std::string> infinite_expiration_times{};
  for (const auto& [control_system_name, combined_name] : map_of_names) {
    (void)control_system_name;
    if (combined_expiration_times.count(combined_name) != 1) {
      combined_expiration_times[combined_name] =
          std::numeric_limits<double>::infinity();
    }
  }

  [[maybe_unused]] const auto combine_expiration_times =
      [&initial_time, &measurements_per_update, &domain_creator, &map_of_names,
       &combined_expiration_times,
       &infinite_expiration_times](const auto& option_holder) {
        const std::string& control_system_name =
            std::decay_t<decltype(option_holder)>::control_system::name();
        const std::string& combined_name = map_of_names[control_system_name];

        auto tuner = option_holder.tuner;
        Tags::detail::initialize_tuner(make_not_null(&tuner), domain_creator,
                                       initial_time, control_system_name);

        const auto& controller = option_holder.controller;
        const DataVector measurement_timescales =
            calculate_measurement_timescales(controller, tuner,
                                             measurements_per_update);
        const double min_measurement_timescale = min(measurement_timescales);

        double initial_expiration_time = function_of_time_expiration_time(
            initial_time, DataVector{1, 0.0},
            DataVector{1, min_measurement_timescale}, measurements_per_update);
        initial_expiration_time = option_holder.is_active
                                      ? initial_expiration_time
                                      : std::numeric_limits<double>::infinity();

        if (initial_expiration_time ==
            std::numeric_limits<double>::infinity()) {
          infinite_expiration_times.insert(control_system_name);
        }

        combined_expiration_times[combined_name] = std::min(
            combined_expiration_times[combined_name], initial_expiration_time);
      };

  EXPAND_PACK_LEFT_TO_RIGHT(combine_expiration_times(option_holders));

  // Set all functions of time for a given measurement to the same expiration
  // time
  for (const auto& [control_system_name, combined_name] : map_of_names) {
    initial_expiration_times[control_system_name] =
        infinite_expiration_times.count(control_system_name) == 1
            ? std::numeric_limits<double>::infinity()
            : combined_expiration_times[combined_name];
  }

  return initial_expiration_times;
}
}  // namespace control_system
