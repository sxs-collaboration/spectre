// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "Time/OptionTags/StepChoosers.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for a vector of ::StepChooser%s
struct StepChoosers : db::SimpleTag {
  using type =
      std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>;
  using option_tags = tmpl::list<::OptionTags::StepChoosers>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& step_choosers) {
    return deserialize<type>(serialize<type>(step_choosers).data());
  }
};
}  // namespace Tags
