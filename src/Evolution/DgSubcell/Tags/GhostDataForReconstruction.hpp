// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"

namespace evolution::dg::subcell::Tags {
/// The ghost data used for reconstructing the solution on the interfaces
/// between elements
///
/// The `FixedHashMap` stores a `evolution::dg::subcell::GhostData` which stores
/// both local and received neighbor ghost data. Even though subcell doesn't use
/// it's local ghost data during reconstruction, we will need to store it when
/// using Charm++ messages.
template <size_t Dim>
struct GhostDataForReconstruction : db::SimpleTag {
  using type =
      FixedHashMap<maximum_number_of_neighbors(Dim),
                   std::pair<Direction<Dim>, ElementId<Dim>>, GhostData,
                   boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>;
};
}  // namespace evolution::dg::subcell::Tags
