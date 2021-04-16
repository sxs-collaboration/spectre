// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace Xcts {
/// \brief Types of conformal geometries for the XCTS equations
enum class Geometry {
  /// Euclidean (flat) manifold with Cartesian coordinates, i.e. the conformal
  /// metric has components \f$\bar{\gamma}_{ij} = \delta_{ij}\f$ in these
  /// coordinates and thus all Christoffel symbols vanish:
  /// \f$\bar{\Gamma}^i_{jk}=0\f$.
  FlatCartesian,
  /// The conformal geometry is either curved or employs curved coordinates, so
  /// non-vanishing Christoffel symbols must be taken into account.
  Curved
};
}  // namespace Xcts
