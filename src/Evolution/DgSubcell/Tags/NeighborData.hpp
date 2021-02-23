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
#include "Evolution/DgSubcell/NeighborData.hpp"

namespace evolution::dg::subcell::Tags {
/// The neighbor data for reconstruction and the RDMP troubled-cell indicator.
///
/// This also holds the self-information for the RDMP at the time level `t^n`
/// (the candidate is at `t^{n+1}`), with id `ElementId::external_boundary_id`
/// and `Direction::lower_xi()` as the (arbitrary and meaningless)
/// direction.
template <size_t Dim>
struct NeighborDataForReconstructionAndRdmpTci : db::SimpleTag {
  using type =
      FixedHashMap<maximum_number_of_neighbors(Dim) + 1,
                   std::pair<Direction<Dim>, ElementId<Dim>>, NeighborData,
                   boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>;
};
}  // namespace evolution::dg::subcell::Tags
