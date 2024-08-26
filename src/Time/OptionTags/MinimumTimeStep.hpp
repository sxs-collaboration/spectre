// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Tags.hpp"
#include "Options/String.hpp"

namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
/// \brief The minimum step size without triggering an error
struct MinimumTimeStep {
  using type = double;
  static constexpr Options::String help =
      "The minimum step size without triggering an error";
  static type lower_bound() { return 0.; }
  using group = evolution::OptionTags::Group;
};
}  // namespace OptionTags
