// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearSolver/Gmres.hpp"

#include <blaze/math/Column.h>
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/DynamicVector.h>
#include <blaze/math/Submatrix.h>
#include <blaze/math/Subvector.h>
#include <blaze/math/lapack/trsv.h>
#include <complex>

#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace LinearSolver::gmres::detail {

namespace {

template <typename ValueType>
void update_givens_rotation(const gsl::not_null<ValueType*> givens_cosine,
                            const gsl::not_null<double*> givens_sine,
                            const ValueType rho, const double sigma) {
  if (UNLIKELY(rho == 0.)) {
    *givens_cosine = 0.;
    *givens_sine = 1.;
  } else {
    const double tmp = sqrt(square(abs(rho)) + square(sigma));
    *givens_cosine = rho / tmp;
    *givens_sine = sigma / tmp;
  }
}

template <typename ValueType, typename Arg>
void apply_and_update_givens_rotation(
    Arg argument,
    const gsl::not_null<blaze::DynamicVector<double>*> givens_sine_history,
    const gsl::not_null<blaze::DynamicVector<ValueType>*> givens_cosine_history,
    const size_t iteration) {
  const size_t k = iteration + 1;
  for (size_t i = 0; i < k - 1; ++i) {
    const ValueType tmp = (*givens_cosine_history)[i] * argument[i] +
                          (*givens_sine_history)[i] * argument[i + 1];
    argument[i + 1] = (*givens_cosine_history)[i] * argument[i + 1] -
                      (*givens_sine_history)[i] * argument[i];
    argument[i] = tmp;
  }
  // Note: argument[k] is real, since it is a normalization
  ASSERT(equal_within_roundoff(std::imag(argument[k]), 0.),
         "Normalization is not real: " << argument[k]);
  update_givens_rotation(make_not_null(&(*givens_cosine_history)[k - 1]),
                         make_not_null(&(*givens_sine_history)[k - 1]),
                         argument[k - 1], std::real(argument[k]));
  argument[k - 1] = (*givens_cosine_history)[k - 1] * argument[k - 1] +
                    (*givens_sine_history)[k - 1] * argument[k];
  argument[k] = 0.;
}

}  // namespace

template <typename ValueType>
void solve_minimal_residual(
    const gsl::not_null<blaze::DynamicMatrix<ValueType>*>
        orthogonalization_history,
    const gsl::not_null<blaze::DynamicVector<ValueType>*> residual_history,
    const gsl::not_null<blaze::DynamicVector<double>*> givens_sine_history,
    const gsl::not_null<blaze::DynamicVector<ValueType>*> givens_cosine_history,
    const size_t iteration) {
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

template <typename ValueType>
blaze::DynamicVector<ValueType> minimal_residual_vector(
    const blaze::DynamicMatrix<ValueType>& orthogonalization_history,
    const blaze::DynamicVector<ValueType>& residual_history) {
  const size_t length = orthogonalization_history.columns();
  blaze::DynamicVector<ValueType> minres =
      blaze::subvector(residual_history, 0, length);
  blaze::trsv(blaze::submatrix(orthogonalization_history, 0, 0, length, length),
              minres, 'U', 'N', 'N');
  return minres;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(r, data)                                          \
  template void solve_minimal_residual(                               \
      gsl::not_null<blaze::DynamicMatrix<DTYPE(data)>*>,              \
      gsl::not_null<blaze::DynamicVector<DTYPE(data)>*>,              \
      gsl::not_null<blaze::DynamicVector<double>*>,                   \
      gsl::not_null<blaze::DynamicVector<DTYPE(data)>*>, size_t);     \
  template blaze::DynamicVector<DTYPE(data)> minimal_residual_vector( \
      const blaze::DynamicMatrix<DTYPE(data)>&,                       \
      const blaze::DynamicVector<DTYPE(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, std::complex<double>))

#undef DTYPE
#undef INSTANTIATE

}  // namespace LinearSolver::gmres::detail
