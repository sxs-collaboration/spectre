// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "ControlSystem/Controller.hpp"
#include "ControlSystem/InitialExpirationTimes.hpp"
#include "ControlSystem/Tags.hpp"
#include "ControlSystem/Tags/FunctionsOfTimeInitialize.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
template <class Metavariables, typename ControlSystem>
struct ControlComponent;
/// \endcond

namespace control_system {
/*!
 * \ingroup ControlSystemGroup
 * \brief Calculate the measurement timescale based on the damping timescale,
 * update fraction, and DerivOrder of the control system
 *
 * The update timescale is \f$\tau_\mathrm{update} = \alpha_\mathrm{update}
 * \tau_\mathrm{damp}\f$ where \f$\tau_\mathrm{damp}\f$ is the damping timescale
 * (from the TimescaleTuner) and \f$\alpha_\mathrm{update}\f$ is the update
 * fraction (from the controller). For an Nth order control system, the averager
 * requires at least N measurements in order to perform its finite
 * differencing to calculate the derivatives of the control error. This implies
 * that the largest the measurement timescale can be is \f$\tau_\mathrm{m} =
 * \tau_\mathrm{update} / N\f$. To ensure that we have sufficient measurements,
 * we calculate the measurement timescales as \f$\tau_\mathrm{m} =
 * \tau_\mathrm{update} / (N+1)\f$
 */
template <size_t DerivOrder>
DataVector calculate_measurement_timescales(
    const ::Controller<DerivOrder>& controller, const ::TimescaleTuner& tuner) {
  return tuner.current_timescale() * controller.get_update_fraction() *
         (1.0 / static_cast<double>(DerivOrder + 1));
}

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// \brief The measurement timescales associated with
/// domain::Tags::FunctionsOfTime.
///
/// Each function of time associated with a control system has a corresponding
/// set of timescales here, represented as `PiecewisePolynomial<0>` with the
/// same components as the function itself.
struct MeasurementTimescales : db::SimpleTag {
  using type = std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;

  static constexpr bool pass_metavariables = true;

  template <typename Metavariables>
  using option_tags = typename detail::OptionList<
      Metavariables, false,
      ::detail::has_override_functions_of_time_v<Metavariables>>::type;

  /// This version of create_from_options is used if the metavariables
  /// defined a constexpr bool `override_functions_of_time` and it is `true`,
  /// and there are control systems in the metavariables
  template <typename Metavariables, typename... OptionHolders>
  static type create_from_options(
      const std::optional<std::string>& function_of_time_file,
      const std::map<std::string, std::string>& function_of_time_name_map,
      const double initial_time, const double initial_time_step,
      const OptionHolders&... option_holders) {
    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        timescales{};
    const auto initial_expiration_times =
        control_system::initial_expiration_times(
            initial_time, initial_time_step, option_holders...);

    [[maybe_unused]] const auto construct_measurement_timescales =
        [&timescales, &initial_time, &initial_time_step,
         &initial_expiration_times, &function_of_time_file,
         &function_of_time_name_map](const auto& option_holder) {
          // This check is intentionally inside the lambda so that it will not
          // trigger for domains without control systems.
          if (initial_time_step <= 0.0) {
            ERROR(
                "Control systems can only be used in forward-in-time "
                "evolutions.");
          }

          const auto& controller = option_holder.controller;
          const std::string& name =
              std::decay_t<decltype(option_holder)>::control_system::name();
          const auto& tuner = option_holder.tuner;

          DataVector measurement_timescales =
              calculate_measurement_timescales(controller, tuner);
          // At a minimum, we can only measure once a time step with GTS.
          for (size_t i = 0; i < measurement_timescales.size(); i++) {
            measurement_timescales[i] =
                std::max(initial_time_step, measurement_timescales[i]);
          }

          double expr_time = initial_expiration_times.at(name);

          // If we are reading this function of time in from a file, set the
          // measurement timescales to be infinity, and to never expire.
          if (function_of_time_file.has_value()) {
            for (const auto& [spec_name, spectre_name] :
                 function_of_time_name_map) {
              (void)spec_name;
              if (name == spectre_name) {
                expr_time = std::numeric_limits<double>::infinity();

                for (size_t i = 0; i < measurement_timescales.size(); i++) {
                  measurement_timescales[i] =
                      std::numeric_limits<double>::infinity();
                }
              }
            }
          }

          timescales.emplace(
              name,
              std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
                  initial_time, std::array{std::move(measurement_timescales)},
                  expr_time));
        };

    EXPAND_PACK_LEFT_TO_RIGHT(construct_measurement_timescales(option_holders));

    return timescales;
  }

  /// This version of create_from_options is used if the metavariables did not
  /// define a constexpr bool `override_functions_of_time` or it did define it
  /// and it is `false`, and the metavariables did define control systems
  template <typename Metavariables, typename... OptionHolders>
  static type create_from_options(const double initial_time,
                                  const double initial_time_step,
                                  const OptionHolders&... option_holders) {
    return MeasurementTimescales::create_from_options<Metavariables>(
        std::nullopt, {}, initial_time, initial_time_step, option_holders...);
  }

  // We don't define a `create_from_options` for when there aren't control
  // systems in the metavariables because it doesn't make sense to have
  // measurement timescales without control systems
};
}  // namespace Tags
}  // namespace control_system
