// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace domain {

/*!
 * \ingroup ComputationalDomainGroup
 * \brief Geometry of a block in the computational domain
 */
enum class BlockGeometry {
  /// A logical cube that can be deformed by coordinate maps. In each direction
  /// it has zero or one block neighbor.
  Cube,
  /// A spherical shell. It only has block neighbors in the radial direction. In
  /// each radial direction it has either zero block neighbors (at the
  /// boundary), one block neighbor (another spherical shell), or multiple block
  /// neighbors (cubes deformed to wedges, 4 in 2D or 6 in 3D).
  SphericalShell
};

std::ostream& operator<<(std::ostream& os, BlockGeometry shell_type);

}  // namespace domain
