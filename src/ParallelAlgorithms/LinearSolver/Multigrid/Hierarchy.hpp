// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <unordered_set>
#include <vector>

/// \cond
template <size_t Dim>
struct ElementId;
/// \endcond

namespace LinearSolver::multigrid {

/*!
 * \brief Coarsen the initial refinement levels of all blocks in the domain
 *
 * Simply decrement the refinement level uniformly over the entire domain.
 * Doesn't do anything for blocks that are already fully coarsened, so if the
 * return value equals the input argument the entire domain is fully coarsened.
 * Decrementing the refinement level means combining two elements into one,
 * thereby halving the number of elements per dimension.
 *
 * \tparam Dim The spatial dimension of the domain
 * \param initial_refinement_levels The refinement level in each block of the
 * domain and in every dimension.
 * \return std::vector<std::array<size_t, Dim>> The coarsened refinement levels
 * by decrementing every entry in `initial_refinement_levels` unless it is
 * already zero.
 */
template <size_t Dim>
std::vector<std::array<size_t, Dim>> coarsen(
    std::vector<std::array<size_t, Dim>> initial_refinement_levels) noexcept;

/*!
 * \brief The element covering the `child_id` on the coarser grid.
 *
 * \tparam Dim The spatial dimension of the domain
 * \param child_id The ID of an element on the finer grid
 * \return ElementId<Dim> The ID of the element on the coarser grid that
 * covers the `child_id`. This parent element covers at most two child elements
 * per dimension.
 */
template <size_t Dim>
ElementId<Dim> parent_id(const ElementId<Dim>& child_id) noexcept;

/*!
 * \brief The elements covering the `parent_id` on the finer grid.
 *
 * \tparam Dim The spatial dimension of the domain
 * \param parent_id The ID of an element on the coarser grid
 * \param children_refinement_levels The refinement level of the finer grid in
 * this block
 * \return std::unordered_set<ElementId<Dim>> The IDs of the elements on the
 * finer grid that cover the `parent_id`. Returns an empty set if the
 * `parent_id` is already on the finest grid. Else, returns at least one child
 * (if the grids have the same refinement levels) and at most
 * \f$2^\mathrm{Dim}\f$ children (if the grid is finer in every dimension).
 */
template <size_t Dim>
std::unordered_set<ElementId<Dim>> child_ids(
    const ElementId<Dim>& parent_id,
    const std::array<size_t, Dim>& children_refinement_levels) noexcept;

}  // namespace LinearSolver::multigrid
