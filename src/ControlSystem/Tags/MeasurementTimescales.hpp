// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <memory>
#include <string>
#include <unordered_map>

#include "ControlSystem/Controller.hpp"
#include "ControlSystem/InitialExpirationTimes.hpp"
#include "ControlSystem/Tags.hpp"
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
 * requires at least N measurements in order to perfom its finitite differencing
 * to calculate the derivatives of the control error. This implies that the
 * largest the measurement timescale can be is \f$\tau_\mathrm{m} =
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
  using option_tags = tmpl::flatten<
      tmpl::list<::OptionTags::InitialTime, ::OptionTags::InitialTimeStep,
                 control_system::inputs<tmpl::transform<
                     tmpl::filter<typename Metavariables::component_list,
                                  tt::is_a_lambda<ControlComponent, tmpl::_1>>,
                     tmpl::bind<tmpl::back, tmpl::_1>>>>>;

  template <typename Metavariables, typename... OptionHolders>
  static type create_from_options(const double initial_time,
                                  const double initial_time_step,
                                  const OptionHolders&... option_holders) {
    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        timescales{};
    const auto initial_expiration_times =
        control_system::initial_expiration_times(
            initial_time, initial_time_step, option_holders...);

    [[maybe_unused]] const auto construct_measurement_timescales =
        [&timescales, &initial_time, &initial_time_step,
         &initial_expiration_times](const auto& option_holder) {
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

          timescales.emplace(
              name,
              std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
                  initial_time, std::array{std::move(measurement_timescales)},
                  initial_expiration_times.at(name)));
        };

    EXPAND_PACK_LEFT_TO_RIGHT(construct_measurement_timescales(option_holders));

    return timescales;
  }
};
}  // namespace Tags
}  // namespace control_system
