// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace Poisson {
/// \brief Types of background geometries for the Poisson equation
enum class Geometry {
  /// Euclidean (flat) manifold with Cartesian coordinates, i.e. the metric has
  /// components \f$\gamma_{ij} = \delta_{ij}\f$ in these coordinates and thus
  /// all Christoffel symbols vanish: \f$\Gamma^i_{jk}=0\f$.
  FlatCartesian,
  /// The manifold is either curved or employs curved coordinates, so
  /// non-vanishing Christoffel symbols must be taken into account.
  Curved
};
}  // namespace Poisson
