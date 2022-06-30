// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "ControlSystem/InitialExpirationTimes.hpp"
#include "ControlSystem/Tags.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/ReadSpecPiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/OptionTags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
template <class Metavariables, typename ControlSystem>
struct ControlComponent;
/// \endcond

namespace control_system::Tags {
namespace detail {

template <typename Metavariables, bool NeedDomainCreator,
          bool HasOverrideCubicFunctionsOfTime>
struct OptionList {
  using option_holders = control_system::inputs<
      tmpl::transform<tmpl::filter<typename Metavariables::component_list,
                                   tt::is_a<ControlComponent, tmpl::_1>>,
                      tmpl::bind<tmpl::back, tmpl::_1>>>;
  static constexpr bool metavars_has_control_systems =
      tmpl::size<option_holders>::value > 0;

  using type = tmpl::flatten<tmpl::list<
      tmpl::conditional_t<NeedDomainCreator,
                          tmpl::list<domain::OptionTags::DomainCreator<
                              Metavariables::volume_dim>>,
                          tmpl::list<>>,
      tmpl::conditional_t<
          Metavariables::override_functions_of_time,
          tmpl::list<
              domain::FunctionsOfTime::OptionTags::FunctionOfTimeFile,
              domain::FunctionsOfTime::OptionTags::FunctionOfTimeNameMap>,
          tmpl::list<>>,
      tmpl::conditional_t<
          metavars_has_control_systems,
          tmpl::list<::OptionTags::InitialTime, ::OptionTags::InitialTimeStep,
                     option_holders>,
          tmpl::list<>>>>;
};

template <typename Metavariables, bool NeedDomainCreator>
struct OptionList<Metavariables, NeedDomainCreator, false> {
  using option_holders = control_system::inputs<
      tmpl::transform<tmpl::filter<typename Metavariables::component_list,
                                   tt::is_a<ControlComponent, tmpl::_1>>,
                      tmpl::bind<tmpl::back, tmpl::_1>>>;
  static constexpr bool metavars_has_control_systems =
      tmpl::size<option_holders>::value > 0;

  using type = tmpl::flatten<tmpl::list<
      tmpl::conditional_t<NeedDomainCreator,
                          tmpl::list<domain::OptionTags::DomainCreator<
                              Metavariables::volume_dim>>,
                          tmpl::list<>>,
      tmpl::conditional_t<
          metavars_has_control_systems,
          tmpl::list<::OptionTags::InitialTime, ::OptionTags::InitialTimeStep,
                     option_holders>,
          tmpl::list<>>>>;
};

// Check that all control systems are actually controlling a function of
// time, and that the expiration times have been set appropriately. If there
// exists a control system that isn't controlling a function of time, or the
// expiration times were set improperly, this is an error and we shouldn't
// continue.
void check_expiration_time_consistency(
    const std::unordered_map<std::string, double>& initial_expiration_times,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time);
}  // namespace detail

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
  using option_tags = typename detail::OptionList<
      Metavariables, true,
      ::detail::has_override_functions_of_time_v<Metavariables>>::type;

  /// This version of create_from_options is used if the metavariables defined a
  /// constexpr bool `override_functions_of_time` and it is `true`, and there
  /// are control systems in the metavariables
  template <typename Metavariables, typename... OptionHolders>
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator,
      const std::optional<std::string>& function_of_time_file,
      const std::map<std::string, std::string>& function_of_time_name_map,
      const double initial_time, const double initial_time_step,
      const OptionHolders&... option_holders) {
    const auto initial_expiration_times =
        control_system::initial_expiration_times(
            initial_time, initial_time_step, option_holders...);

    // We need to check the expiration times before we replace functions of
    // time (if we're going to do so) so we can ensure a proper domain creator
    // was chosen from options.
    auto functions_of_time =
        domain_creator->functions_of_time(initial_expiration_times);

    detail::check_expiration_time_consistency(initial_expiration_times,
                                              functions_of_time);

    if (function_of_time_file.has_value()) {
      domain::FunctionsOfTime::override_functions_of_time(
          make_not_null(&functions_of_time), *function_of_time_file,
          function_of_time_name_map);
    }

    return functions_of_time;
  }

  /// This version of create_from_options is used if the metavariables defined a
  /// constexpr bool `override_functions_of_time` and it is `true`, but there
  /// are no control systems in the metavariables
  template <typename Metavariables>
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator,
      const std::optional<std::string>& function_of_time_file,
      const std::map<std::string, std::string>& function_of_time_name_map) {
    // Just use 0.0 for initial time and time step. Since there are no
    // control systems, these values won't be used anyways
    return FunctionsOfTimeInitialize::create_from_options<Metavariables>(
        domain_creator, function_of_time_file, function_of_time_name_map, 0.0,
        0.0);
  }

  /// This version of create_from_options is used if the metavariables did not
  /// define a constexpr bool `override_functions_of_time` or it did define it
  /// and it is `false`, and the metavariables did define control systems
  template <typename Metavariables, typename... OptionHolders>
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator,
      const double initial_time, const double initial_time_step,
      const OptionHolders&... option_holders) {
    return FunctionsOfTimeInitialize::create_from_options<Metavariables>(
        domain_creator, std::nullopt, {}, initial_time, initial_time_step,
        option_holders...);
  }

  /// This version of create_from_options is used if the metavariables did not
  /// define a constexpr bool `override_functions_of_time` or it did define it
  /// and it is `false`, and the metavariables did not define control systems
  template <typename Metavariables>
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator) {
    return domain_creator->functions_of_time();
  }
};
}  // namespace control_system::Tags
