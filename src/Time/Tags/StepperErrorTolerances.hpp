// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Time/StepperErrorTolerances.hpp"

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for the stepper error tolerances.
template <typename Tag>
struct StepperErrorTolerances : db::PrefixTag, db::SimpleTag {
  static std::string name() {
    return "StepperErrorTolerances(" + db::tag_name<Tag>() + ")";
  }
  using type = std::optional<::StepperErrorTolerances>;
  using tag = Tag;
};
}  // namespace Tags
