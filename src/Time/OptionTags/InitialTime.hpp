// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Tags.hpp"
#include "Options/String.hpp"

namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
/// \brief The time at which to start the simulation
struct InitialTime {
  using type = double;
  static constexpr Options::String help = {
      "The time at which the evolution is started."};
  static type suggested_value() { return 0.0; }
  using group = evolution::OptionTags::Group;
};
}  // namespace OptionTags
