// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "Evolution/Tags.hpp"
#include "Options/String.hpp"
#include "Time/StepChoosers/StepChooser.hpp"

namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
struct StepChoosers {
  static constexpr Options::String help{"Limits on LTS step size"};
  using type =
      std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>;
  static size_t lower_bound_on_size() { return 1; }
  using group = evolution::OptionTags::Group;
};
}  // namespace OptionTags
