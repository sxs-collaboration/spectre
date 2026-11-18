// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

/// \cond
template <size_t VolumeDim>
class Element;
template <size_t Dim>
class Mesh;
/// \endcond

namespace evolution::dg::subcell {
/*!
 * \brief Whether subcell is allowed in the element, ignoring any restriction
 * at external boundaries.
 *
 * Subcell is not allowed if the DG mesh does not support subcell (e.g. a
 * non-hypercube topology like a spherical shell), if the element is in one of
 * the `only_dg_block_ids`, or if any of its neighbors is. Elements for which
 * this is `false` should skip the TCI and always stay on the DG grid.
 */
template <size_t Dim>
bool subcell_allowed_in_element(const Element<Dim>& element,
                                const Mesh<Dim>& dg_mesh,
                                const std::vector<size_t>& only_dg_block_ids);

/*!
 * \brief Whether the element starts the evolution on the subcell (FD) grid.
 *
 * This is the case if subcell is allowed in the element (see
 * `subcell_allowed_in_element()`) and the element is either not on an external
 * boundary or subcell is enabled at external boundaries.
 *
 * With `AlwaysUseSubcells`, exactly these elements stay on the subcell grid for
 * the whole evolution.
 */
template <size_t Dim>
bool starts_on_subcell_grid(const Element<Dim>& element,
                            const Mesh<Dim>& dg_mesh,
                            const std::vector<size_t>& only_dg_block_ids,
                            bool subcell_enabled_at_external_boundary);
}  // namespace evolution::dg::subcell
