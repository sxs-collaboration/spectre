// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
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
/// \brief The number of finite-difference subcell grid points per dimension as
/// a function of the number of DG grid points.
///
/// If `Auto` then \f$2N-1\f$ FD grid points are used for \f$N\f$ DG grid
/// points. Otherwise three integers \f$[a, b, c]\f$ are given and
/// \f$aN^2 + bN + c\f$ FD grid points are used. It is an error if this is
/// smaller than \f$N\f$.
///
/// \note This lives outside the `DgSubcell` library for the same reason as
/// `OnlyDgBlocksAndGroups` (see
/// `evolution::dg::Tags::NumberOfSubcellPointsCoefficients`).
struct NumberOfSubcellPoints {
  using type = Options::Auto<std::array<int, 3>>;
  static constexpr Options::String help = {
      "The number of FD grid points per dimension for N DG grid points. "
      "'Auto' uses 2N-1. Otherwise a list of three integers [a, b, c] gives "
      "a*N^2 + b*N + c, which must be at least N."};
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
      "instead of the number of DG grid points for elements that start on "
      "the subcell grid, i.e. that are in subcell-capable blocks and that "
      "neither border a DG-only block nor are on an external boundary where "
      "subcell is disabled. This better reflects the higher computational "
      "cost of running the FD scheme on an element compared to running DG on "
      "it. Requires ElementDistribution to be NumGridPoints. This option is "
      "only available for executables with DG-subcell support."};
  using group = Parallel::OptionTags::Parallelization;
};
}  // namespace evolution::dg::OptionTags
