// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Tags.hpp"
#include "Options/String.hpp"

namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
/// \brief The initial time step taken by the time stepper. This may be
/// overridden by an adaptive stepper
struct InitialTimeStep {
  using type = double;
  static constexpr Options::String help =
      "The initial time step, before local stepping adjustment";
  using group = evolution::OptionTags::Group;
};
}  // namespace OptionTags
