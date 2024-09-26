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

namespace evolution::dg::subcell::Tags {
/*!
 * \brief Holds the volume Mesh used by each neighbor for communicating subcell
 * ghost data.
 *
 * \details This will be the subcell Mesh unless an Element doing DG has only
 * neighbors also doing DG
 */
template <size_t Dim>
struct MeshForGhostData : db::SimpleTag {
  using type = DirectionalIdMap<Dim, ::Mesh<Dim>>;
};
}  // namespace evolution::dg::subcell::Tags
