// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the function find_generalized_eigenvalues_arpack.

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
 * \brief Solves a generalized eigenvalue problem (matrix_a v_a=lambda_a
 * matrix_b v_a) using
 * Arpack. Shift invert transform is used to convert the generalized eigenvalue
 * problem into a standard eigenvalue problem and then Arpack's dnaupd and
 * dneupd routines are called.
 *
 * This function uses the Arpack routine dnaupd and dneupd
 * (https://www.caam.rice.edu/software/ARPACK/UG/ug.html) to solve (matrix_a v_a
 * =lambda_a matrix_b v_a) for the generalized eigenvalues lambda_a and
 * corresponding eigenvectors v_a. There is no routine in Arpack that can
 * directly work with matrices if matrix_b is non-symmetric. To solve such
 * problems we need to transform (matrix_a v_a =lambda_a matrix_b v_a)  into
 * (matrix_a matrix_b^{-1} v_a =lambda_a v_a) i.e. a generalized eigenvalue
 * problem into a standard eigenvalue problem. We also use shift invert
 * transform for faster convergence because we want to find the smallest
 * eigenvalues and Arpack is much faster in finding the largest eigenvalues.
 * Using shift inverse transform means that we need the action of the matrix
 * (matrix_a - sigma*matrix_b)^{-1} * matrix_b on a vector. To get the action of
 * (matrix_a - sigma*matrix_b)^{-1} Lapack's LU solver routine is used.
 * Documentation for the dnaupd function can be found in the arpack guide page
 * 125 or at the link: (https://www.caam.rice.edu/software/ARPACK/UG/
 * node137.html#SECTION001220000000000000000) Some addition information about
 * the choice of algorithm is present on the pages 91 and 36.
 *
 * INPUT PARAMETERS :
 *
 * eigenvalues_real_part: Vector of size number_of_eigenvalues_to_find that will
 * store the real part of the eigenvalues.
 *
 * eigenvalues_imaginary_part: Vector of size number_of_eigenvalues_to_find that
 * will store the imaginary part of the eigenvalues.
 *
 * eigenvectors: Stores the eigenvectors. If the eigenvalues are real then, the
 * eigenvector corresponding to the eigenvalue at the index i is stored in the
 * indices i*number_of_rows to i*(1+number_of_rows) where number_of_rows is the
 * number of rows of the square matrix matrix_a.
 *
 * matrix_a: The square matrix matrix_a in the equation (matrix_a v_a = lambda_a
 * matrix_b v_a), this matrix is overwritten when the function is called.
 * matrix_a should be stored in the column major order.
 *
 * matrix_b: The square matrix matrix_b in the equation (matrix_a v_a = lambda_a
 * matrix_b v_a). matrix_b should be stored in the column major order.
 *
 * number_of_eigenvalues_to_find: The number of eigenvalues we want to find.
 * In our case we know that the eigenvalues are going to be real so we need the
 * smallest three so we can set the value to this variable to 3. But, if for a
 * problem it is not guaranteed that all the eigenvalues we are interested in
 * are going to be real, then it is safer to set this parameter to a larger
 * value. For example, let us say that the eigenvalues of a system are 1,2,3 +
 * i, 3 - i,4,5 and we ask for the three smallest magnitude eigenvalues. Because
 * the size of the eigenvector variable is (number_of_eigenvalues_to_find *
 * number of rows in matrix_a ) there is no space to store both the real and the
 * imaginary part of the complex eigenvector corresponding to the eigenvalue
 * 3+i. Thus, it is better to ask for more eigenvalues than required if it is
 * not guaranteed that all the eigenvalues are going to be real so that there is
 * enough space to store the complex eigenvectors.
 *
 * sigma: The real part of the sigma used in shift invert transform. We know
 * that our eigenvalues are going to be real so we do not need a complex shift
 * invert transform. Shift invert operation transforms (matrix_a v_a = lambda_a
 * matrix_b v_a) into ( (matrix_a - sigma*matrix_b )^{-1} * matrix_b v_a = nu_a
 * v_a), here nu_a is related to lambda_a via nu_a = (lambda_a - sigma)^{-1}.
 *
 * which_eigenvalues_to_find: A string which tells Arpack which eigenvalues to
 * search for. Possible values are `LA' and `SA' for the algebraically largest
 * and smallest eigenvalues, `LM' and `SM' for the eigenvalues of largest or
 * smallest magnitude, and `BE' for the simultaneous computation of the
 * eigenvalues at both ends of the spectrum. There are four other modes that
 * choose the eigenvalues with largest/smallest real/imaginary parts (page 78,
 * https://github.com/m-reuter/arpackpp/blob/master/doc/arpackpp.pdf), but are
 * not mentioned in the Arpack pdf guide because it is a bit outdated.
 *
 * The following comments are only relevant if one is interested in
 * using this function to solve eigenvalue problems with complex eigenvectors.
 * Because, in this particular case we know in advance that the eigenvectors are
 * going to be real they are here only for future reference. If the eigenvalues
 * are complex, then the real and imaginary parts of the corresponding
 * eigenvector are stored consecutively in eigenvectors( real part starting from
 * 'i*number_of_rows' and imaginary part from 'i*number_of_rows+1' where
 * number_of_rows is the number of rows in the square matrix matrix_a). The next
 * eigenvector and eigenvalue will be the complex conjugate of these. NOTE: on
 * the page 36 of the Arpack manual it is says that "When the eigenvectors
 * corresponding to a complex comjugate pair of eigenvalues are computed, the
 * vector corresponding to the eigenvalue with positive imaginary part is stored
 * with real and imaginary parts in consecutive columns ...". In my limited
 * testing this does not seem to hold true when shift invert tranforms are used.
 * Even the scipy code seems to follow the procedure described in the previous
 * comment:
 * (https://github.com/scipy/scipy/blob/f397c8e17ac235316da6f1a10692b2604b6d6ec4
 * /scipy/sparse/linalg/eigen/arpack/arpack.py#L801)
 */
void find_generalized_eigenvalues_arpack(
    DataVector& eigenvalues_real_part, DataVector& eigenvalues_imaginary_part,
    Matrix& eigenvectors, Matrix& matrix_a, Matrix& matrix_b,
    const size_t number_of_eigenvalues_to_find, const double sigma,
    const std::string which_eigenvalues_to_find) noexcept;
