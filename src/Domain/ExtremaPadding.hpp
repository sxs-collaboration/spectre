// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
/// \endcond

namespace domain {
/*!
 * \brief Estimate of how much a function can exceed its extrema over the
 * grid points of an element anywhere inside the element, given its
 * `grid_point_values`.
 *
 * The extrema of a coordinate (or of the distance from a point) over the grid
 * points of a curved element can be smaller than over the whole element, e.g.
 * because the center of a curved face is not a grid point. Between
 * neighboring grid points, a function exceeds the linear interpolant between
 * them by at most \f$h^2 \max|f''|/8\f$, where \f$h\f$ is the spacing in the
 * logical coordinate and \f$f''\f$ is the second derivative with respect to
 * it. The padding is the sum of this bound over the logical dimensions, with
 * the second derivatives estimated by divided differences, so it vanishes for
 * functions that are linear in the logical coordinates (e.g. the coordinates of
 * elements with straight edges). It should be used as a cheap test whether a
 * point can be in the element.
 *
 * \param mesh the mesh of the element, which should have grid points on the
 * element boundaries (e.g. a DG mesh)
 * \param grid_point_values the values at the grid points of `mesh` of a
 * function that is smooth in the element, e.g. a coordinate or the distance
 * from a point
 */
double extrema_padding(const Mesh<3>& mesh,
                       const DataVector& grid_point_values);
}  // namespace domain
