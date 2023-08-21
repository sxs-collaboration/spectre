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
#include "ControlSystem/Controller.hpp"
#include "ControlSystem/ExpirationTimes.hpp"
#include "ControlSystem/Tags/OptionTags.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
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
/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// \brief The measurement timescales associated with
/// domain::Tags::FunctionsOfTime.
///
/// Each function of time associated with a control system has a corresponding
/// set of timescales here, represented as `PiecewisePolynomial<0>` with the
/// same components as the function itself.
///
/// If the control system isn't active, then the measurement timescale and
/// expiration time are `std::numeric_limits<double>::infinity()`.
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

    [[maybe_unused]] const auto construct_measurement_timescales =
        [&timescales, &initial_time, &initial_time_step, &domain_creator,
         &measurements_per_update](const auto& option_holder) {
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
          auto tuner = option_holder.tuner;
          Tags::detail::initialize_tuner(make_not_null(&tuner), domain_creator,
                                         initial_time, name);

          DataVector measurement_timescales = calculate_measurement_timescales(
              controller, tuner, measurements_per_update);

          double expr_time = measurement_expiration_time(
              initial_time, DataVector{measurement_timescales.size(), 0.0},
              measurement_timescales, measurements_per_update);

          // If the control system isn't active, set measurement timescale and
          // expiration time to be infinity.
          if (not option_holder.is_active) {
            expr_time = std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < measurement_timescales.size(); i++) {
              measurement_timescales[i] =
                  std::numeric_limits<double>::infinity();
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
};
}  // namespace control_system::Tags
