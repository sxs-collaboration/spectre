// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function find_generalized_eigenvalues.

#pragma once

/// \cond
class DataVector;
class Matrix;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Solve the standard eigenvalue problem for a non-symmetric matrix using
 * Arpack. Can be used to solve generalized eigenvalue problem by transforming
 * it into a standard eigenvalue problem.
 *
 * This function uses the Arpack++ routine ARNonSymStdEig
 * (https://github.com/m-reuter/arpackpp/tree/2.3.0/doc)
 * to solve the generalized eigenvalue problem \f$A v_a =\lambda_a B v_a \f$
 * for the generalized eigenvalues \f$\lambda_a\f$ and corresponding
 * eigenvectors \f$v_a\f$. There is no routine in Arpack that can directly work
 * with matrices if B is non-symmetric. To solve such problems we need to
 * transform \f$A v_a =\lambda_a B v_a \f$ into \f$A B^{-1} v_a =\lambda_a
 * v_a\f$.
 * `matrix_a` and `matrix_b` are each a
 * `Matrix`; they correspond to square matrices \f$A\f$ and \f$B\f$ that are the
 * same dimension \f$N\f$. `eigenvalues_real_part` is a `DataVector` of size
 * \f$N\f$ that will store the real parts of the eigenvalues,
 * `eigenvalues_imaginary_part` is a `DataVector` of size \f$N\f$ that will
 * store the imaginary parts of the eigenvalues. The real and imaginary parts of
 * the eigenvectors are stored in the `eigenvector_real` and `eigenvector_imag`
 * respectively, second index is for iterating over the eigenvector and first
 * index is for iterating over the components of that eigenvector(same as
 * LAPACK). `number_of_evals_to_find` is the number of eigenvalues/eigenvectors
 * that we want to find. `sigma` is the value of the shift parameter in the
 * shift invert transform. `which` is the parameters that dictates which
 * eigenvalues are found. Note: Arpack is much faster when it has to find the
 * largest eigenvalues which is why if we need to find the smallest eigenvalues
 * it much faster to do a shift-invert transform first.
 *
 */
void find_generalized_eigenvalues_arpack(
    DataVector& eigenvalues_real_part, DataVector& eigenvalues_imaginary_part,
    Matrix& eigenvectors, Matrix& matrix_a, Matrix& matrix_b,
    const size_t number_of_eigenvalues_to_find, const double sigma,
    const std::string which_eigenvalues_to_find) noexcept;
