// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"

namespace evolution::dg::subcell::Tags {
/// The neighbor data for reconstruction.
template <size_t Dim>
struct NeighborDataForReconstruction : db::SimpleTag {
  using type =
      FixedHashMap<maximum_number_of_neighbors(Dim),
                   std::pair<Direction<Dim>, ElementId<Dim>>,
                   std::vector<double>,
                   boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>;
};
}  // namespace evolution::dg::subcell::Tags
