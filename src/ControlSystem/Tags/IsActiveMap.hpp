// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <map>
#include <optional>
#include <string>
#include <unordered_map>

#include "ControlSystem/Metafunctions.hpp"
#include "ControlSystem/Tags/OptionTags.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <typename Metavariables, typename ControlSystem>
struct ControlComponent;
namespace domain::FunctionsOfTime::OptionTags {
struct FunctionOfTimeFile;
struct FunctionOfTimeNameMap;
}  // namespace domain::FunctionsOfTime::OptionTags
/// \endcond

namespace control_system::Tags {
namespace detail {
template <typename... OptionHolders>
std::unordered_map<std::string, bool> create_is_active_map(
    const OptionHolders&... option_holders) {
  std::unordered_map<std::string, bool> result{};

  [[maybe_unused]] const auto add_to_result =
      [&result](const auto& option_holder) {
        using control_system =
            typename std::decay_t<decltype(option_holder)>::control_system;
        result[control_system::name()] = option_holder.is_active;
      };

  EXPAND_PACK_LEFT_TO_RIGHT(add_to_result(option_holders));

  return result;
}
}  // namespace detail

/*!
 * \ingroup DataBoxTagsGroup
 * \ingroup ControlSystemGroup
 * \brief Tag that holds a map between control system name and whether that
 * control system is active. Can be used in the GlobalCache.
 *
 * This effectively lets us choose control systems at runtime. The OptionHolder
 * has an option for whether the control system is active.
 */
struct IsActiveMap : db::SimpleTag {
  using type = std::unordered_map<std::string, bool>;
  static constexpr bool pass_metavariables = true;

  template <typename Component>
  struct system {
    using type = typename Component::control_system;
  };

  template <typename Metavariables>
  using option_tags = control_system::inputs<tmpl::transform<
      metafunctions::all_control_components<Metavariables>, system<tmpl::_1>>>;

  template <typename Metavariables, typename... OptionHolders>
  static type create_from_options(const OptionHolders&... option_holders) {
    return detail::create_is_active_map(option_holders...);
  }
};
}  // namespace control_system::Tags
