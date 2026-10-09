// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/ConcentricShell.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace domain {
namespace {
// Largest variation of the radius over a sphere of grid points, relative to
// the largest radius, for which a shell is considered concentric. This allows
// for roundoff in the coordinate maps.
constexpr double relative_tolerance_for_concentric_shell = 1.0e-12;
}  // namespace

std::optional<std::pair<double, double>> concentric_shell_radial_extent(
    const Mesh<3>& mesh, const DataVector& radii) {
  ASSERT(radii.size() == mesh.number_of_grid_points(),
         "The mesh has " << mesh.number_of_grid_points()
                         << " grid points but there are " << radii.size()
                         << " radii.");
  const bool is_s2_shell =
      (mesh.basis(0) == Spectral::Basis::Legendre or
       mesh.basis(0) == Spectral::Basis::Chebyshev) and
      mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto and
      mesh.basis(1) == Spectral::Basis::SphericalHarmonic and
      mesh.basis(2) == Spectral::Basis::SphericalHarmonic;
  if (not is_s2_shell) {
    return std::nullopt;
  }
  const double tolerance = relative_tolerance_for_concentric_shell * max(radii);
  // Grid points with the same radial index must all have the same radius.
  // The radial index varies fastest.
  const size_t radial_extent = mesh.extents(0);
  for (size_t radial_index = 0; radial_index < radial_extent; ++radial_index) {
    double min_radius = radii[radial_index];
    double max_radius = radii[radial_index];
    for (size_t s = radial_index; s < radii.size(); s += radial_extent) {
      min_radius = std::min(min_radius, radii[s]);
      max_radius = std::max(max_radius, radii[s]);
    }
    if (max_radius - min_radius > tolerance) {
      return std::nullopt;
    }
  }
  return std::pair{min(radii) - tolerance, max(radii) + tolerance};
}
}  // namespace domain
