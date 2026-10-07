// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <utility>

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
/// \endcond

namespace domain {
/*!
 * \brief The range of `radii` over the element if the element is a
 * spherical-harmonic shell concentric with the center the radii are measured
 * from, `std::nullopt` otherwise.
 *
 * The element is considered a concentric shell if `mesh` is a
 * spherical-harmonic shell with Gauss-Lobatto quadrature in the radial
 * dimension and the grid points with the same radial index all have the same
 * radius, up to a small tolerance for roundoff. A point is in the element
 * exactly if its radius is in the returned range, which is padded by the
 * tolerance. This holds for any (e.g. time-dependent) map that keeps the
 * shell spherical and concentric, such as rotations and expansions about the
 * center.
 *
 * \param mesh the mesh of the element
 * \param radii the distances of the grid points of `mesh` from the center
 */
std::optional<std::pair<double, double>> concentric_shell_radial_extent(
    const Mesh<3>& mesh, const DataVector& radii);
}  // namespace domain
