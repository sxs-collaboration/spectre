// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"

/// \cond
template <size_t Dim, typename T>
class DirectionalIdMap;
template <size_t Dim>
class Mesh;
/// \endcond

namespace domain::Tags {
/*!
 * \brief Holds the mesh of each neighboring element.
 *
 * This knowledge can be used to determine the geometry of mortars between
 * elements. It is kept up to date by AMR.
 *
 * For DG-FD hybrid methods this is necessary to determine what numerical method
 * the neighbor is using. This knowledge can be used for optimizing code.
 */
template <size_t Dim>
struct NeighborMesh : db::SimpleTag {
  using type = DirectionalIdMap<Dim, ::Mesh<Dim>>;
};
}  // namespace domain::Tags
