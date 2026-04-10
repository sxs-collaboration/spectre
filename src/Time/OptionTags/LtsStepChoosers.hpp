// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <vector>

#include "Evolution/Tags.hpp"
#include "Options/String.hpp"
#include "Time/StepChoosers/StepChooser.hpp"

namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
struct LtsStepChoosers {
  static constexpr Options::String help{
      "Limits on the LTS step size.  If the list is empty, the step:slab "
      "ratio will not be changed."};
  using type =
      std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>;
  using group = evolution::OptionTags::Group;
};
}  // namespace OptionTags
