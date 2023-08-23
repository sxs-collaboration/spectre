// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "ControlSystem/CalculateMeasurementTimescales.hpp"
#include "ControlSystem/CombinedName.hpp"
#include "ControlSystem/Controller.hpp"
#include "ControlSystem/ExpirationTimes.hpp"
#include "ControlSystem/Metafunctions.hpp"
#include "ControlSystem/Tags/OptionTags.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Time/OptionTags/InitialTime.hpp"
#include "Time/OptionTags/InitialTimeStep.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
template <class Metavariables, typename ControlSystem>
struct ControlComponent;
/// \endcond

namespace control_system::Tags {
/*!
 * \ingroup DataBoxTagsGroup
 * \ingroup ControlSystemGroup
 * \brief The measurement timescales associated with
 * domain::Tags::FunctionsOfTime.
 *
 * We group measurement timescales by
 * the `control_system::protocols::Measurement` that their corresponding control
 * systems use. This is because one measurement may be used to update many
 * functions of time. Each group of functions of time associated with a
 * particular measurement has a corresponding timescale here, represented as
 * `PiecewisePolynomial<0>` with a single entry. That single entry is the
 * minimum of all `control_system::calculate_measurement_timescales` for every
 * control system in that group.
 *
 * The name of a measurement timescale is calculated using
 * `control_system::system_to_combined_names` for every group of control systems
 * with the same measurement.
 *
 * If all control systems that use the same measurement aren't active, then the
 * measurement timescale and expiration time are
 * `std::numeric_limits<double>::infinity()`.
 */
struct MeasurementTimescales : db::SimpleTag {
  using type = std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;

  template <typename Metavariables>
  using option_holders = control_system::inputs<
      tmpl::transform<tmpl::filter<typename Metavariables::component_list,
                                   tt::is_a<ControlComponent, tmpl::_1>>,
                      tmpl::bind<tmpl::back, tmpl::_1>>>;

  static constexpr bool pass_metavariables = true;

  template <typename Metavariables>
  using option_tags = tmpl::push_front<
      option_holders<Metavariables>,
      control_system::OptionTags::MeasurementsPerUpdate,
      domain::OptionTags::DomainCreator<Metavariables::volume_dim>,
      ::OptionTags::InitialTime, ::OptionTags::InitialTimeStep>;

  template <typename Metavariables, typename... OptionHolders>
  static type create_from_options(
      const int measurements_per_update,
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator,
      const double initial_time, const double initial_time_step,
      const OptionHolders&... option_holders) {
    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        timescales{};

    using control_systems =
        tmpl::list<typename OptionHolders::control_system...>;

    // First string is name of each control system. Second string is combination
    // of control systems with same measurement
    std::unordered_map<std::string, std::string> map_of_names =
        system_to_combined_names<control_systems>();

    // Initialize all measurements to infinity so we can take the min later
    std::unordered_map<std::string, double> min_measurement_timescales{};
    std::unordered_map<std::string, double> expiration_times{};
    for (const auto& [control_system_name, combined_name] : map_of_names) {
      (void)control_system_name;
      if (min_measurement_timescales.count(combined_name) != 1) {
        min_measurement_timescales[combined_name] =
            std::numeric_limits<double>::infinity();
        expiration_times[combined_name] =
            std::numeric_limits<double>::infinity();
      }
    }

    [[maybe_unused]] const auto combine_measurement_timescales =
        [&initial_time, &initial_time_step, &domain_creator,
         &measurements_per_update, &map_of_names, &min_measurement_timescales,
         &expiration_times](const auto& option_holder) {
          // This check is intentionally inside the lambda so that it will not
          // trigger for domains without control systems.
          if (initial_time_step <= 0.0) {
            ERROR(
                "Control systems can only be used in forward-in-time "
                "evolutions.");
          }

          const std::string& control_system_name =
              std::decay_t<decltype(option_holder)>::control_system::name();
          const std::string& combined_name = map_of_names[control_system_name];

          auto tuner = option_holder.tuner;
          Tags::detail::initialize_tuner(make_not_null(&tuner), domain_creator,
                                         initial_time, control_system_name);

          const auto& controller = option_holder.controller;
          DataVector measurement_timescales = calculate_measurement_timescales(
              controller, tuner, measurements_per_update);

          double min_measurement_timescale = min(measurement_timescales);

          // If the control system isn't active, set measurement timescale and
          // expiration time to be infinity.
          if (not option_holder.is_active) {
            min_measurement_timescale = std::numeric_limits<double>::infinity();
          }

          min_measurement_timescales[combined_name] =
              std::min(min_measurement_timescales[combined_name],
                       min_measurement_timescale);

          if (min_measurement_timescales[combined_name] !=
              std::numeric_limits<double>::infinity()) {
            const double expiration_time = measurement_expiration_time(
                initial_time, DataVector{1_st, 0.0},
                DataVector{1_st, min_measurement_timescales[combined_name]},
                measurements_per_update);
            expiration_times[combined_name] =
                std::min(expiration_times[combined_name], expiration_time);
          }
        };

    EXPAND_PACK_LEFT_TO_RIGHT(combine_measurement_timescales(option_holders));

    for (const auto& [combined_name, min_measurement_timescale] :
         min_measurement_timescales) {
      double expiration_time = expiration_times[combined_name];
      timescales.emplace(
          combined_name,
          std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
              initial_time,
              std::array{DataVector{1, min_measurement_timescale}},
              expiration_time));
    }

    return timescales;
  }
};
}  // namespace control_system::Tags
