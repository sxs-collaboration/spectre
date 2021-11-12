// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "ControlSystem/Tags.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/OptionTags.hpp"
#include "Time/Tags.hpp"
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
                                  tt::is_a_lambda<ControlComponent, tmpl::_1>>,
                     tmpl::bind<tmpl::back, tmpl::_1>>>>>;

  template <typename Metavariables, typename... OptionHolders>
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator,
      const double initial_time, const double initial_time_step,
      const OptionHolders&... option_holders) {
    std::unordered_map<std::string, double> initial_expiration_times{};
    [[maybe_unused]] const auto gather_initial_expiration_times =
        [&initial_expiration_times, &initial_time,
         &initial_time_step](const auto& option_holder) {
          const auto& controller = option_holder.controller;
          const std::string& name =
              std::decay_t<decltype(option_holder)>::control_system::name();
          const auto& tuner = option_holder.tuner;

          const double update_fraction = controller.get_update_fraction();
          const double curr_timescale = min(tuner.current_timescale());
          const double initial_expiration_time =
              update_fraction * curr_timescale;
          initial_expiration_times[name] =
              initial_time +
              std::max(initial_time_step, initial_expiration_time);
        };
    EXPAND_PACK_LEFT_TO_RIGHT(gather_initial_expiration_times(option_holders));
    // Until the domain creator infrastructure is modified to take control
    // system information into account when constructing the functions of time,
    // update them retroactively. This is not how this will normally be done,
    // there just needs to be a place holder until the domain creators have been
    // changed.
    auto functions_of_time = domain_creator->functions_of_time();
    for (auto& [name, expr_time] : initial_expiration_times) {
      const double curr_expr_time =
          functions_of_time.at(name)->time_bounds()[1];
      functions_of_time.at(name)->reset_expiration_time(
          std::max(curr_expr_time, expr_time));
    }
    return functions_of_time;
  }
};
}  // namespace control_system::Tags
