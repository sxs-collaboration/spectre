// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/ParallelAlgorithms/LinearSolver/DistributedLinearSolverAlgorithmTestHelpers.hpp"

namespace DistributedLinearSolverAlgorithmTestHelpers {

blaze::DynamicMatrix<double> combine_matrix_slices(
    const std::vector<blaze::DynamicMatrix<double>>& matrix_slices) {
  const size_t num_slices = matrix_slices.size();
  const size_t num_cols_per_slice = matrix_slices.begin()->columns();
  const size_t total_num_points = num_slices * num_cols_per_slice;
  blaze::DynamicMatrix<double> full_matrix(total_num_points, total_num_points);
  for (size_t i = 0; i < num_slices; ++i) {
    blaze::submatrix(full_matrix, 0, i * num_cols_per_slice, total_num_points,
                     num_cols_per_slice) = gsl::at(matrix_slices, i);
  }
  return full_matrix;
}

}  // namespace DistributedLinearSolverAlgorithmTestHelpers
