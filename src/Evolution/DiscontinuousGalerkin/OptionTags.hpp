// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <vector>

#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/OptionsGroup.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"

namespace evolution::dg::OptionTags {
/// \ingroup OptionTagsGroup
/// \brief A list of block and group names on which to never do subcell.
///
/// Set to `None` to allow subcell in all blocks.
///
/// \note This lives outside the `DgSubcell` library so that it can also be
/// read by code that must not depend on `DgSubcell`, such as the element
/// distribution (see `evolution::dg::Tags::OnlyDgBlockIds`)
struct OnlyDgBlocksAndGroups {
  using type =
      Options::Auto<std::vector<std::string>, Options::AutoLabel::None>;
  static constexpr Options::String help = {
      "A list of block and group names on which to never do subcell.\n"
      "Set to 'None' to not restrict where FD can be used."};
  using group = ::dg::OptionTags::DiscontinuousGalerkinGroup;
};
}  // namespace evolution::dg::OptionTags
