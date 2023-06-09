// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/OptionTags.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell {
namespace OptionTags {
struct ActiveGrid {
  using type = evolution::dg::subcell::ActiveGrid;
  static constexpr Options::String help = {
      "The type of the active grid. Either 'Dg' or 'Subcell'."};
  using group = SpatialDiscretization::OptionTags::SpatialDiscretizationGroup;
};
}  // namespace OptionTags
namespace Tags {

/// The grid currently used for the DG-subcell evolution on the element.
struct ActiveGrid : db::SimpleTag {
  using type = evolution::dg::subcell::ActiveGrid;
  using option_tags =
      tmpl::list<evolution::dg::subcell::OptionTags::ActiveGrid>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type option) { return option; }
};
}  // namespace Tags
}  // namespace evolution::dg::subcell
