// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for the stepper error measure.
template <typename Tag>
struct StepperError : db::PrefixTag, db::SimpleTag {
  static std::string name() {
    return "StepperError(" + db::tag_name<Tag>() + ")";
  }
  using type = typename Tag::type;
  using tag = Tag;
};
}  // namespace Tags
