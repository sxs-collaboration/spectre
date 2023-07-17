// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Tags.hpp"
#include "Options/String.hpp"

namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
/// \brief The initial slab size
struct InitialSlabSize {
  using type = double;
  static constexpr Options::String help = "The initial slab size";
  static type lower_bound() { return 0.; }
  using group = evolution::OptionTags::Group;
};
}  // namespace OptionTags
