// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"

namespace LinearSolver::Schwarz {

/// Identifies a subdomain region that overlaps with another element
template <size_t Dim>
using OverlapId = std::pair<Direction<Dim>, ElementId<Dim>>;

/// Data structure that can store the `ValueType` on each possible overlap of an
/// element-centered subdomain with its neighbors. Overlaps are identified by
/// their `OverlapId`.
template <size_t Dim, typename ValueType>
using OverlapMap =
    FixedHashMap<maximum_number_of_neighbors(Dim), OverlapId<Dim>, ValueType,
                 boost::hash<OverlapId<Dim>>>;

/*!
 * \brief The number of points that an overlap extends into the `volume_extent`
 *
 * In a dimension where an element has `volume_extent` points, the overlap
 * extent is the largest number under these constraints:
 *
 * - It is at most `max_overlap`.
 * - It is smaller than the `volume_extent`.
 *
 * This means the overlap extent is always smaller than the `volume_extent`. The
 * reason for this constraint is that we define the _width_ of the overlap as
 * the element-logical coordinate distance from the face of the element to the
 * first collocation point _outside_ the overlap extent. Therefore, even an
 * overlap region that covers the full element in width does not include the
 * collocation point on the opposite side of the element.
 *
 * Here's a few notes on the definition of the overlap extent and width:
 *
 * - A typical smooth weighting function goes to zero at the overlap width, so
 *   if the grid points located at the overlap width were included in the
 *   subdomain, their solutions would not contribute to the weighted sum of
 *   subdomain solutions.
 * - Defining the overlap width as the distance to the first point _outside_ the
 *   overlap extent makes it non-zero even for a single point of overlap into a
 *   Gauss-Lobatto grid (which has points located at the element face).
 * - Boundary contributions for many (but not all) discontinuous Galerkin
 *   schemes on Gauss-Lobatto grids are limited to the grid points on the
 *   element face, e.g. for a DG operator that is pre-multiplied by the mass
 *   matrix, or one where boundary contributions are lifted using the diagonal
 *   mass-matrix approximation. Not including the grid points facing away from
 *   the subdomain in the overlap allows to ignore that face altogether in the
 *   subdomain operator.
 */
size_t overlap_extent(size_t volume_extent, size_t max_overlap) noexcept;

/*!
 * \brief Total number of grid points in an overlap region that extends
 * `overlap_extent` points into the `volume_extents` from either side in the
 * `overlap_dimension`
 *
 * The overlap region has `overlap_extent` points in the `overlap_dimension`,
 * and `volume_extents` points in the other dimensions. The number of grid
 * points returned by this function is the product of these extents.
 */
template <size_t Dim>
size_t overlap_num_points(const Index<Dim>& volume_extents,
                          size_t overlap_extent,
                          size_t overlap_dimension) noexcept;

/*!
 * \brief Width of an overlap extending `overlap_extent` points into the
 * `collocation_points` from either side.
 *
 * The "width" of an overlap is the element-logical coordinate distance from the
 * element boundary to the first collocation point outside the overlap region in
 * the overlap dimension, i.e. the dimension perpendicular to the element face.
 * See `LinearSolver::Schwarz::overlap_extent` for details.
 *
 * This function assumes the `collocation_points` are mirrored around 0.
 */
double overlap_width(size_t overlap_extent,
                     const DataVector& collocation_points) noexcept;

}  // namespace LinearSolver::Schwarz
