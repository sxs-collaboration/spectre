// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Filtering.hpp"

#include <cmath>

#include "DataStructures/Matrix.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"

namespace Spectral {
namespace filtering {
Matrix exponential_filter(const Mesh<1>& mesh, const double alpha,
                          const unsigned half_power) noexcept {
  if (UNLIKELY(mesh.number_of_grid_points() == 1)) {
    return Matrix(1, 1, 1.0);
  }
  const Matrix& nodal_to_modal = Spectral::nodal_to_modal_matrix(mesh);
  const Matrix& modal_to_nodal = Spectral::modal_to_nodal_matrix(mesh);
  Matrix filter_matrix(mesh.number_of_grid_points(),
                       mesh.number_of_grid_points(), 0.0);
  const double order = mesh.number_of_grid_points() - 1.0;
  for (size_t i = 0; i < mesh.number_of_grid_points(); ++i) {
    filter_matrix(i, i) = exp(-alpha * pow(i / order, 2 * half_power));
  }
  return modal_to_nodal * filter_matrix * nodal_to_modal;
}
}  // namespace filtering
}  // namespace Spectral
