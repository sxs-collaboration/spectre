// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearSolver/Gmres.hpp"

#include <blaze/math/Column.h>
#include <blaze/math/Submatrix.h>
#include <blaze/math/Subvector.h>
#include <blaze/math/lapack/trsv.h>

#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace LinearSolver {
namespace gmres_detail {

namespace {

void update_givens_rotation(const gsl::not_null<double*> givens_cosine,
                            const gsl::not_null<double*> givens_sine,
                            const double rho, const double sigma) noexcept {
  if (UNLIKELY(rho == 0.)) {
    *givens_cosine = 0.;
    *givens_sine = 1.;
  } else {
    const double tmp = sqrt(square(rho) + square(sigma));
    *givens_cosine = abs(rho) / tmp;
    *givens_sine = *givens_cosine * sigma / rho;
  }
}

template <typename Arg>
void apply_and_update_givens_rotation(
    Arg argument, const gsl::not_null<DenseVector<double>*> givens_sine_history,
    const gsl::not_null<DenseVector<double>*> givens_cosine_history,
    const size_t iteration) noexcept {
  const size_t k = iteration + 1;
  for (size_t i = 0; i < k - 1; ++i) {
    const double tmp = (*givens_cosine_history)[i] * argument[i] +
                       (*givens_sine_history)[i] * argument[i + 1];
    argument[i + 1] = (*givens_cosine_history)[i] * argument[i + 1] -
                      (*givens_sine_history)[i] * argument[i];
    argument[i] = tmp;
  }
  update_givens_rotation(make_not_null(&(*givens_cosine_history)[k - 1]),
                         make_not_null(&(*givens_sine_history)[k - 1]),
                         argument[k - 1], argument[k]);
  argument[k - 1] = (*givens_cosine_history)[k - 1] * argument[k - 1] +
                    (*givens_sine_history)[k - 1] * argument[k];
  argument[k] = 0.;
}

}  // namespace

void solve_minimal_residual(
    const gsl::not_null<DenseMatrix<double>*> orthogonalization_history,
    const gsl::not_null<DenseVector<double>*> residual_history,
    const gsl::not_null<DenseVector<double>*> givens_sine_history,
    const gsl::not_null<DenseVector<double>*> givens_cosine_history,
    const size_t iteration) noexcept {
  residual_history->resize(iteration + 2);
  givens_sine_history->resize(iteration + 1);
  givens_cosine_history->resize(iteration + 1);
  // Givens-rotate the `orthogonalization_history` to eliminate its lower-right
  // entry, making it an upper triangular matrix.
  // Thus, we iteratively update the QR decomposition of the Hessenberg matrix
  // that is built through orthogonalization of the Krylov basis vectors.
  apply_and_update_givens_rotation(
      blaze::column(*orthogonalization_history, iteration), givens_sine_history,
      givens_cosine_history, iteration);
  // Also Givens-rotate the `residual_history`. The vector excluding its last
  // entry represents the accumulated Givens-rotation applied to the vector
  // `(initial_residual, 0, 0, ...)` and the last entry represents the
  // remaining residual.
  (*residual_history)[iteration + 1] =
      -(*givens_sine_history)[iteration] * (*residual_history)[iteration];
  (*residual_history)[iteration] =
      (*givens_cosine_history)[iteration] * (*residual_history)[iteration];
}

DenseVector<double> minimal_residual_vector(
    const DenseMatrix<double>& orthogonalization_history,
    const DenseVector<double>& residual_history) noexcept {
  const size_t length = orthogonalization_history.columns();
  DenseVector<double> minres = blaze::subvector(residual_history, 0, length);
  blaze::trsv(blaze::submatrix(orthogonalization_history, 0, 0, length, length),
              minres, 'U', 'N', 'N');
  return minres;
}

}  // namespace gmres_detail
}  // namespace LinearSolver
