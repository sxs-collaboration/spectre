// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "ControlSystem/InitialExpirationTimes.hpp"
#include "ControlSystem/Tags.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/OptionTags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
template <class Metavariables, typename ControlSystem>
struct ControlComponent;
/// \endcond

namespace control_system::Tags {
/// \ingroup ControlSystemGroup
/// The FunctionsOfTime initialized from a DomainCreator, initial time
/// step, and control system OptionHolders.
struct FunctionsOfTimeInitialize : domain::Tags::FunctionsOfTime,
                                   db::SimpleTag {
  using type = std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;

  static constexpr bool pass_metavariables = true;

  static std::string name() { return "FunctionsOfTime"; }

  template <typename Metavariables>
  using option_tags = tmpl::flatten<
      tmpl::list<domain::OptionTags::DomainCreator<Metavariables::volume_dim>,
                 ::OptionTags::InitialTime, ::OptionTags::InitialTimeStep,
                 control_system::inputs<tmpl::transform<
                     tmpl::filter<typename Metavariables::component_list,
                                  tt::is_a<ControlComponent, tmpl::_1>>,
                     tmpl::bind<tmpl::back, tmpl::_1>>>>>;

  template <typename Metavariables, typename... OptionHolders>
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator,
      const double initial_time, const double initial_time_step,
      const OptionHolders&... option_holders) {
    const std::unordered_map<std::string, double> initial_expiration_times =
        control_system::initial_expiration_times(
            initial_time, initial_time_step, option_holders...);

    auto functions_of_time =
        domain_creator->functions_of_time(initial_expiration_times);

    // Check that all control systems are actually controlling a function of
    // time, and that the expiration times have been set appropriately. If there
    // exists a control system that isn't controlling a function of time, or the
    // expiration times were set improperly, this is an error and we shouldn't
    // continue.
    for (const auto& [name, expr_time] : initial_expiration_times) {
      if (functions_of_time.count(name) == 0) {
        ERROR(
            "The control system '"
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

    return functions_of_time;
  }
};
}  // namespace control_system::Tags
