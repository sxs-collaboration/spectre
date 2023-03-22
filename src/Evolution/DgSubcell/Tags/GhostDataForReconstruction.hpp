// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"

namespace evolution::dg::subcell::Tags {
/// The ghost data from neighboring elements used for reconstructing the
/// solution on the interfaces between elements
template <size_t Dim>
struct GhostDataForReconstruction : db::SimpleTag {
  using type =
      FixedHashMap<maximum_number_of_neighbors(Dim),
                   std::pair<Direction<Dim>, ElementId<Dim>>, DataVector,
                   boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>;
};
}  // namespace evolution::dg::subcell::Tags
