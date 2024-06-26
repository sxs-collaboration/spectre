// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <optional>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Time/StepperErrorEstimate.hpp"

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for the stepper error measures.
template <typename Tag>
struct StepperErrors : db::PrefixTag, db::SimpleTag {
  static std::string name() {
    return "StepperErrors(" + db::tag_name<Tag>() + ")";
  }
  using type = std::array<std::optional<StepperErrorEstimate>, 2>;
  using tag = Tag;
};
}  // namespace Tags
