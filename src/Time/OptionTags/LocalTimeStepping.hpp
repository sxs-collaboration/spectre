// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Tags.hpp"
#include "Options/String.hpp"
#include "Time/LtsMode.hpp"

namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
/// \brief Local time-stepping mode for the evolution
struct LocalTimeStepping {
  using type = ::LtsMode;
  static constexpr Options::String help =
      "Local time-stepping mode for the evolution";
  using group = evolution::OptionTags::Group;
};
}  // namespace OptionTags
