// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "ControlSystem/ExpirationTimes.hpp"
#include "ControlSystem/Tags/IsActiveMap.hpp"
#include "ControlSystem/Tags/OptionTags.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Time/OptionTags/InitialTime.hpp"
#include "Time/OptionTags/InitialTimeStep.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
template <class Metavariables, typename ControlSystem>
struct ControlComponent;
/// \endcond

namespace control_system::Tags {
namespace detail {
// Check that all control systems are actually controlling a function of
// time, and that the expiration times have been set appropriately. If there
// exists a control system that isn't controlling a function of time, or the
// expiration times were set improperly, this is an error and we shouldn't
// continue.
void check_expiration_time_consistency(
    const std::unordered_map<std::string, double>& initial_expiration_times,
    const std::unordered_map<std::string, bool>& is_active_map,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time);
}  // namespace detail

/// \ingroup ControlSystemGroup
/// The FunctionsOfTime initialized from a DomainCreator, initial time, and
/// control system OptionHolders.
struct FunctionsOfTimeInitialize : domain::Tags::FunctionsOfTime,
                                   db::SimpleTag {
  using type = std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;

  static constexpr bool pass_metavariables = true;

  static std::string name() { return "FunctionsOfTime"; }

  template <typename Metavariables>
  using option_holders = control_system::inputs<
      tmpl::transform<tmpl::filter<typename Metavariables::component_list,
                                   tt::is_a<ControlComponent, tmpl::_1>>,
                      tmpl::bind<tmpl::back, tmpl::_1>>>;

  template <typename Metavariables>
  static constexpr bool metavars_has_control_systems =
      tmpl::size<option_holders<Metavariables>>::value > 0;

  template <typename Metavariables>
  using option_tags = tmpl::push_front<
      tmpl::conditional_t<
          metavars_has_control_systems<Metavariables>,
          tmpl::flatten<tmpl::list<
              control_system::OptionTags::MeasurementsPerUpdate,
              ::OptionTags::InitialTime, option_holders<Metavariables>>>,
          tmpl::list<>>,
      domain::OptionTags::DomainCreator<Metavariables::volume_dim>>;

  /// This version of create_from_options is used if the metavariables did
  /// define control systems
  template <typename Metavariables, typename... OptionHolders>
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator,
      const int measurements_per_update,
     const double initial_time, const OptionHolders&... option_holders) {
    const auto initial_expiration_times =
        control_system::initial_expiration_times(
            initial_time, measurements_per_update, domain_creator,
            option_holders...);

    // We need to check the expiration times so we can ensure a proper domain
    // creator was chosen from options.
    auto functions_of_time =
        domain_creator->functions_of_time(initial_expiration_times);

    const auto is_active_map = detail::create_is_active_map(option_holders...);

    detail::check_expiration_time_consistency(initial_expiration_times,
                                              is_active_map, functions_of_time);

    return functions_of_time;
  }

  /// This version of create_from_options is used if the metavariables did not
  /// define control systems
  template <typename Metavariables>
  static type create_from_options(
     const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator) {
    return domain_creator->functions_of_time();
  }
};
}  // namespace control_system::Tags
