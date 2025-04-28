// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <iosfwd>

namespace evolution::dg {
/// \brief Label for a neighboring Element (or Block) that determines how
/// information is exchanged between neighboring Elements
///
/// \details The specific label is determined by the relationship between the
/// block logical coordinates of the neighboring Elements (Blocks) for the
/// points on the interface between them.  In two cases there is a simple
/// relationship between the coordinates:
/// - CopyProject: in this case the block logical coordinates are identical.
///   Therefore a DataVector can be either copied (if the Mesh on each side of
///   the interface are at the same points) or projected to the Mesh of the
///   neighbor.
/// - OrientCopyProject: in this case the block logical coordinates are related
///   by a discrete rotation (represented by an OrientationMap). Therefore a
///   DataVector can be reoriented (with the OrientationMap), and then either
///   copied or projected to the Mesh of the neighbor.
///
/// In the following cases, there is no simple relationship between the block
/// logical coordinates.  Therefore a DataVector must be interpolated to the
/// points of the neighboring Mesh.  The cases differ in which Element does the
/// interpolation:
/// - NonconformingBothInterpolate:  in this case both the Element and its
///   neighbor interpolate data to the grid points of each others Mesh.
/// - NonconformingSelfInterpolates:  in this case the Element will receive the
///   neighbor's boundary data and will interpolate it to the Mesh of the
///   Element.  The Element will then need to send boundary correction data to
///   the neighbor.
/// - NonconformingNeighborInterpolates:  in this case the Element send its
///   boundary data to the neighbor who will then interpolate it to its own
///   Mesh. The neighbor will need to send boundary correction data back to the
///   Element.
///
/// The Element and its neighbor will need to use consistent values of this
/// enum:
/// - In the cases CopyProject, OrientCopyProject, and
///   NonconformingBothInterpolate, neighboring elements should agree on the
///   values.
/// - For NonconformingSelfInterpolates and NonconformingNeighborInterpolates
///   neighboring elements should have different values.  These cases should be
///   used when one Element has many neighboring Elements (e.g. when a single
///   spherical shell abuts a cubes sphere).  In this case it should be more
///   efficient for the single element to send its boundary data to its
///   neighbors which then do the interpolation to their meshes.
enum class InterfaceDataPolicy : uint8_t {
  /// default value is uninitialized
  Uninitialized = 0,
  /// Boundary data can be copied or projected to Mesh of neighbor
  CopyProject = 1,
  /// Boundary data should be reoriented, and then copied or projected to Mesh
  /// of neighbor
  OrientCopyProject = 2,
  /// Boundary data should be interpolated to Mesh of neighbor
  NonconformingBothInterpolate = 3,
  /// Neighbor will send boundary data to be interpolated onto the Mesh of this
  /// Element.  Boundary correction data will then need to be sent to the
  /// neighbor.
  NonconformingSelfInterpolates = 4,
  /// Boundary data should be sent to the neighbor, who will interpolate the
  /// data to its own Mesh.  The neighbor will send boundary correction data
  /// back.
  NonconformingNeighborInterpolates = 5
};

/// Output operator for a InterfaceDataPolicy.
std::ostream& operator<<(std::ostream& os, InterfaceDataPolicy value);
}  // namespace evolution::dg
