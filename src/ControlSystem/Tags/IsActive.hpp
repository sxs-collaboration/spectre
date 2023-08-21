// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <map>
#include <optional>
#include <string>

#include "ControlSystem/Tags/OptionTags.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::FunctionsOfTime::OptionTags {
struct FunctionOfTimeFile;
struct FunctionOfTimeNameMap;
}  // namespace domain::FunctionsOfTime::OptionTags
/// \endcond

namespace control_system::Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag to determine if this control system is active.
///
/// This effectively lets us choose control systems at runtime. The OptionHolder
/// has an option for whether the control system is active.
template <typename ControlSystem>
struct IsActive : db::SimpleTag {
  using type = bool;
  static constexpr bool pass_metavariables = false;
  using option_tags =
      tmpl::list<OptionTags::ControlSystemInputs<ControlSystem>>;

  static bool create_from_options(
      const control_system::OptionHolder<ControlSystem>& option_holder) {
    return option_holder.is_active;
  }
};
}  // namespace control_system::Tags
