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
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace evolution::dg::Tags {
/*!
 * \brief Holds the mesh of each neighboring element.
 *
 * This is ultimately necessary to determine what numerical method the neighbor
 * is using. This knowledge can be used for optimizing code when using a DG-FD
 * hybrid method, but might also be useful more generally..
 */
template <size_t Dim>
struct NeighborMesh : db::SimpleTag {
  using type =
      FixedHashMap<maximum_number_of_neighbors(Dim),
                   std::pair<Direction<Dim>, ElementId<Dim>>, Mesh<Dim>,
                   boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>;
};
}  // namespace evolution::dg::Tags
