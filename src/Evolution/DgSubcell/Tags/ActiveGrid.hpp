// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"

namespace evolution::dg::subcell::Tags {
/// The grid currently used for the DG-subcell evolution on the element.
struct ActiveGrid : db::SimpleTag {
  using type = evolution::dg::subcell::ActiveGrid;
};
}  // namespace evolution::dg::subcell::Tags
