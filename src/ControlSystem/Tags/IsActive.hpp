// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <map>
#include <optional>
#include <string>

#include "ControlSystem/Tags/OptionTags.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateHasStaticMemberVariable.hpp"

/// \cond
namespace domain::FunctionsOfTime::OptionTags {
struct FunctionOfTimeFile;
struct FunctionOfTimeNameMap;
}  // namespace domain::FunctionsOfTime::OptionTags
/// \endcond

namespace control_system::Tags {
namespace detail {

CREATE_HAS_STATIC_MEMBER_VARIABLE(override_functions_of_time)
CREATE_HAS_STATIC_MEMBER_VARIABLE_V(override_functions_of_time)

template <typename Metavariables, typename ControlSystem,
          bool HasOverrideFunctionsOfTime>
struct IsActiveOptionList {
  using type = tmpl::append<
      tmpl::list<OptionTags::ControlSystemInputs<ControlSystem>>,
      tmpl::conditional_t<
          Metavariables::override_functions_of_time,
          tmpl::list<
              domain::FunctionsOfTime::OptionTags::FunctionOfTimeFile,
              domain::FunctionsOfTime::OptionTags::FunctionOfTimeNameMap>,
          tmpl::list<>>>;
};

template <typename Metavariables, typename ControlSystem>
struct IsActiveOptionList<Metavariables, ControlSystem, false> {
  using type = tmpl::list<OptionTags::ControlSystemInputs<ControlSystem>>;
};

template <typename ControlSystem>
bool is_control_system_active(
    const control_system::OptionHolder<ControlSystem>& option_holder,
    const std::optional<std::string>& function_of_time_file,
    const std::map<std::string, std::string>& function_of_time_name_map) {
  if (not function_of_time_file.has_value()) {
    // `None` was specified as the option for the file so we aren't replacing
    // anything
    return option_holder.is_active;
  }

  const std::string& name = ControlSystem::name();

  for (const auto& spec_and_spectre_names : function_of_time_name_map) {
    if (spec_and_spectre_names.second == name) {
      return false;
    }
  }

  return option_holder.is_active;
}
}  // namespace detail

/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag to determine if this control system is active.
///
/// This effectively lets us choose control systems at runtime. The OptionHolder
/// has an option for whether the control system is active.
///
/// That option can be overridden if we are overriding the functions of time. If
/// the metavariables specifies `static constexpr bool
/// override_functions_of_time = true`, then this will check the
/// `domain::FunctionsOfTime::OptionTags::FunctionOfTimeFile` option. If the
/// file is defined, it will loop over the map between SpEC and SpECTRE names
/// from `domain::FunctionsOfTime::OptionTags::FunctionOfTimeNameMap`. If the
/// function of time corresponding to this control system is being overriden
/// with data from the file, then this tag will be `false` so the control system
/// doesn't actually update the function of time.
///
/// If the metavariables doesn't specify `override_functions_of_time`, or it is
/// set to `false`, then this control system is active by default so the tag
/// will be `true`.
template <typename ControlSystem>
struct IsActive : db::SimpleTag {
  using type = bool;

  static constexpr bool pass_metavariables = true;
  template <typename Metavariables>
  using option_tags = typename detail::IsActiveOptionList<
      Metavariables, ControlSystem,
      detail::has_override_functions_of_time_v<Metavariables>>::type;

  template <typename Metavariables>
  static bool create_from_options(
      const control_system::OptionHolder<ControlSystem>& option_holder,
      const std::optional<std::string>& function_of_time_file,
      const std::map<std::string, std::string>& function_of_time_name_map) {
    return detail::is_control_system_active(
        option_holder, function_of_time_file, function_of_time_name_map);
  }

  template <typename Metavariables>
  static bool create_from_options(
      const control_system::OptionHolder<ControlSystem>& option_holder) {
    return option_holder.is_active;
  }
};
}  // namespace control_system::Tags
