// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/LinearOperators.hpp"

#include <array>
#include <cstddef>
#include <functional>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace Cce {
void logical_partial_directional_derivative_of_complex(
    const gsl::not_null<ComplexDataVector*> d_u, const ComplexDataVector& u,
    const Mesh<3>& mesh, const size_t dimension_to_differentiate) {
  const auto empty_matrix = Matrix{};
  std::array<std::reference_wrapper<const Matrix>, 3> matrix_array{
    {std::ref(empty_matrix), std::ref(empty_matrix), std::ref(empty_matrix)}};
  gsl::at(matrix_array, dimension_to_differentiate) =
      std::ref(Spectral::differentiation_matrix(
          mesh.slice_through(dimension_to_differentiate)));
  apply_matrices(d_u, matrix_array, u, mesh.extents());
}
}  // namespace Cce
