// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Declares the interfaces for the BLAS used.
///
/// Wrappers are defined to perform casts from different integer types
/// when the natural type in C++ differs from the BLAS argument.

#pragma once

#ifndef SPECTRE_DEBUG
#include <libxsmm.h>
#endif  // ifndef SPECTRE_DEBUG

#include "ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace blas_detail {
extern "C" {
double ddot_(const int& N, const double* X, const int& INCX, const double* Y,
             const int& INCY);

void dgemm_(const char& TRANSA, const char& TRANSB, const int& M, const int& N,
            const int& K, const double& ALPHA, const double* A, const int& LDA,
            const double* B, const int& LDB, const double& BETA,
            const double* C, const int& LDC);

void dgemv_(const char& TRANS, const int& M, const int& N, const double& ALPHA,
            const double* A, const int& LDA, const double* X, const int& INCX,
            const double& BETA, double* Y, const int& INCY);
}  // extern "C"
}  // namespace blas_detail

/*!
 * \ingroup UtilitiesGroup
 * The dot product of two vectors.
 *
 * \param N the length of the vectors.
 * \param X a pointer to the first element of the first vector.
 * \param INCX the stride for the elements of the first vector.
 * \param Y a pointer to the first element of the second vector.
 * \param INCY the stride for the elements of the second vector.
 * \return the dot product of the given vectors.
 */
inline double ddot_(const size_t& N, const double* X, const size_t& INCX,
                    const double* Y, const size_t& INCY) {
  // INCX and INCY are allowed to be negative by BLAS, but we never
  // use them that way.  If needed, they can be changed here, but then
  // code providing values will also have to be changed to int to
  // avoid warnings.
  return blas_detail::ddot_(gsl::narrow_cast<int>(N), X,
                            gsl::narrow_cast<int>(INCX), Y,
                            gsl::narrow_cast<int>(INCY));
}

// @{
/*!
 * \ingroup UtilitiesGroup
 * \brief Perform a matrix-matrix multiplication
 *
 * Perform the matrix-matrix multiplication
 * \f[
 * C = \alpha \mathrm{op}(A) \mathrm{op}(B) + \beta \mathrm{op}(C)
 * \f]
 *
 * where \f$\mathrm{op}(A)\f$ represents either \f$A\f$ or \f$A^{T}\f$
 * (transpose of \f$A\f$).
 *
 * LIBXSMM, which is much faster than BLAS for small matrices, can be called
 * instead of BLAS by passing the template parameter `true`.
 *
 * \param TRANSA either 'N', 'T' or 'C', transposition of matrix A
 * \param TRANSB either 'N', 'T' or 'C', transposition of matrix B
 * \param M Number of rows in \f$\mathrm{op}(A)\f$
 * \param N Number of columns in \f$\mathrm{op}(B)\f$ and \f$\mathrm{op}(C)\f$
 * \param K Number of columns in \f$\mathrm{op}(A)\f$
 * \param ALPHA specifies \f$\alpha\f$
 * \param A Matrix \f$A\f$
 * \param LDA Specifies first dimension of \f$\mathrm{op}(A)\f$
 * \param B Matrix \f$B\f$
 * \param LDB Specifies first dimension of \f$\mathrm{op}(B)\f$
 * \param BETA specifies \f$\beta\f$
 * \param C Matrix \f$C\f$
 * \param LDC Specifies first dimension of \f$\mathrm{op}(C)\f$
 * \tparam UseLibXsmm if `true` then use LIBXSMM
 */
template <bool UseLibXsmm = false>
inline void dgemm_(const char& TRANSA, const char& TRANSB, const size_t& M,
                   const size_t& N, const size_t& K, const double& ALPHA,
                   const double* A, const size_t& LDA, const double* B,
                   const size_t& LDB, const double& BETA, double* C,
                   const size_t& LDC) {
  ASSERT('N' == TRANSA or 'n' == TRANSA or 'T' == TRANSA or 't' == TRANSA or
             'C' == TRANSA or 'c' == TRANSA,
         "TRANSA must be upper or lower case N, T, or C. See the BLAS "
         "documentation for help.");
  ASSERT('N' == TRANSB or 'n' == TRANSB or 'T' == TRANSB or 't' == TRANSB or
             'C' == TRANSB or 'c' == TRANSB,
         "TRANSB must be upper or lower case N, T, or C. See the BLAS "
         "documentation for help.");
  blas_detail::dgemm_(
      TRANSA, TRANSB, gsl::narrow_cast<int>(M), gsl::narrow_cast<int>(N),
      gsl::narrow_cast<int>(K), ALPHA, A, gsl::narrow_cast<int>(LDA), B,
      gsl::narrow_cast<int>(LDB), BETA, C, gsl::narrow_cast<int>(LDC));
}

// libxsmm is disabled in DEBUG builds because backtraces (from, for
// example, FPEs) do not work when the error occurs in libxsmm code.
#ifndef SPECTRE_DEBUG
template <>
inline void dgemm_<true>(const char& TRANSA, const char& TRANSB,
                         const size_t& M, const size_t& N, const size_t& K,
                         const double& ALPHA, const double* A,
                         const size_t& LDA, const double* B, const size_t& LDB,
                         const double& BETA, double* C, const size_t& LDC) {
  ASSERT('N' == TRANSA or 'n' == TRANSA or 'T' == TRANSA or 't' == TRANSA or
             'C' == TRANSA or 'c' == TRANSA,
         "TRANSA must be upper or lower case N, T, or C. See the BLAS "
         "documentation for help.");
  ASSERT('N' == TRANSB or 'n' == TRANSB or 'T' == TRANSB or 't' == TRANSB or
             'C' == TRANSB or 'c' == TRANSB,
         "TRANSB must be upper or lower case N, T, or C. See the BLAS "
         "documentation for help.");
  const auto m = gsl::narrow_cast<int>(M), n = gsl::narrow_cast<int>(N),
             k = gsl::narrow_cast<int>(K), lda = gsl::narrow_cast<int>(LDA),
             ldb = gsl::narrow_cast<int>(LDB), ldc = gsl::narrow_cast<int>(LDC);
  libxsmm_dgemm(&TRANSA, &TRANSB, &m, &n, &k, &ALPHA, A, &lda, B, &ldb, &BETA,
                C, &ldc);
}
#endif  // ifndef SPECTRE_DEBUG
// @}

// @{
/*!
 * \ingroup UtilitiesGroup
 * \brief Perform a matrix-vector multiplication
 *
 * \f[
 * y = \alpha \mathrm{op}(A) x + \beta y
 * \f]
 *
 * where \f$\mathrm{op}(A)\f$ represents either \f$A\f$ or \f$A^{T}\f$
 * (transpose of \f$A\f$).
 *
 * \param TRANS either 'N', 'T' or 'C', transposition of matrix A
 * \param M Number of rows in \f$\mathrm{op}(A)\f$
 * \param N Number of columns in \f$\mathrm{op}(A)\f$
 * \param ALPHA specifies \f$\alpha\f$
 * \param A Matrix \f$A\f$
 * \param LDA Specifies first dimension of \f$\mathrm{op}(A)\f$
 * \param X Vector \f$x\f$
 * \param INCX Specifies the increment for the elements of \f$x\f$
 * \param BETA Specifies \f$\beta\f$
 * \param Y Vector \f$y\f$
 * \param INCY Specifies the increment for the elements of \f$y\f$
 */
inline void dgemv_(const char& TRANS, const size_t& M, const size_t& N,
                   const double& ALPHA, const double* A, const size_t& LDA,
                   const double* X, const size_t& INCX, const double& BETA,
                   double* Y, const size_t& INCY) {
  ASSERT('N' == TRANS or 'n' == TRANS or 'T' == TRANS or 't' == TRANS or
             'C' == TRANS or 'c' == TRANS,
         "TRANS must be upper or lower case N, T, or C. See the BLAS "
         "documentation for help.");
  // INCX and INCY are allowed to be negative by BLAS, but we never
  // use them that way.  If needed, they can be changed here, but then
  // code providing values will also have to be changed to int to
  // avoid warnings.
  blas_detail::dgemv_(TRANS, gsl::narrow_cast<int>(M), gsl::narrow_cast<int>(N),
                      ALPHA, A, gsl::narrow_cast<int>(LDA), X,
                      gsl::narrow_cast<int>(INCX), BETA, Y,
                      gsl::narrow_cast<int>(INCY));
}
//@}
