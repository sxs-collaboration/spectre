// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <vector>

#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/OptionsGroup.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Parallel/Tags/Parallelization.hpp"

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

/// \ingroup OptionTagsGroup
/// \brief Whether to weight subcell-capable elements by their
/// finite-difference grid points when distributing elements.
///
/// See `evolution::dg::Tags::UseSubcellGridPointsForDistribution`.
struct UseSubcellGridPointsForDistribution {
  using type = bool;
  static constexpr Options::String help = {
      "If true, the element weight used for the ElementDistribution is "
      "computed using the number of finite-difference subcell grid points "
      "instead of the number of DG grid points for elements that are "
      "subcell-capable. This better reflects the higher computational cost "
      "of running the FD scheme on an element compared to running DG on it. "
      "Requires ElementDistribution to be NumGridPoints. This option is only "
      "available for executables with DG-subcell support."};
  using group = Parallel::OptionTags::Parallelization;
};
}  // namespace evolution::dg::OptionTags
