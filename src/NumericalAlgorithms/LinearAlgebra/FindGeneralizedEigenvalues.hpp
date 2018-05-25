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
 * \brief Solve the generalized eigenvalue problem for two matrices.
 *
 * This function uses the lapack routine dggev
 * (http://www.netlib.org/lapack/explore-3.1.1-html/dggev.f.html)
 * to solve the
 * generalized eigenvalue problem \f$A v_a =\lambda_a B v_a \f$
 * for the generalized eigenvalues \f$\lambda_a\f$ and corresponding
 * eigenvectors \f$v_a\f$.
 * `matrix_a` and `matrix_b` are each a `Matrix`; they correspond to square
 * matrices \f$A\f$ and \f$B\f$ that are the same dimension \f$N\f$.
 * `eigenvalues_real_part` is a `DataVector` of size \f$N\f$ that
 * will store the real parts of the eigenvalues,
 * `eigenvalues_imaginary_part` is a `DataVector` of size \f$N\f$
 * that will store the imaginary parts of the eigenvalues.
 * Complex eigenvalues always form complex conjugate pairs, and
 * the \f$j\f$ and \f$j+1\f$ eigenvalues will have the forms
 * \f$a+ib\f$ and \f$a-ib\f$, respectively. The eigenvectors
 * are returned as the columns of a square `Matrix` of dimension \f$N\f$
 * called `eigenvectors`. If eigenvalue \f$j\f$ is real, then column \f$j\f$ of
 * `eigenvectors` is
 * the corresponding eigenvector. If eigenvalue \f$j\f$ and \f$j+1\f$ are
 * complex-conjugate pairs, then the eigenvector for
 * eigenvalue \f$j\f$ is (column j) + \f$i\f$ (column j+1), and the
 * eigenvector for eigenvalue \f$j+1\f$ is (column j) - \f$i\f$ (column j+1).
 *
 */
void find_generalized_eigenvalues(
    gsl::not_null<DataVector*> eigenvalues_real_part,
    gsl::not_null<DataVector*> eigenvalues_imaginary_part,
    gsl::not_null<Matrix*> eigenvectors, Matrix matrix_a,
    Matrix matrix_b) noexcept;
