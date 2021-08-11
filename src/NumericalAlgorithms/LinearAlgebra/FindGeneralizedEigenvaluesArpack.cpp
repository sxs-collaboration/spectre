// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearAlgebra/FindGeneralizedEigenvaluesArpack.hpp"

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/DenseVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

// LAPACK routine to do the generalized eigenvalue problem
extern "C" {
void dgetrs_(char* trans, int* n, int* nrhs, double* a, int* lda, int* ipiv,
             double* b, int* ldb, int* info);

void dneupd_(int* rvec, char const* howmny, int const* select, double* dr,
             double* di, double* z, int* ldz, double* sigmar, double* sigmai,
             double* workev, char const* bmat, int* n, char const* which,
             int* nev, double* tol, double* resid, int* ncv, double* v,
             int* ldv, int* iparam, int* ipntr, double* workd, double* workl,
             int* lworkl, int* info);

void dnaupd_(int* ido, char const* bmat, int* n, char const* which, int* nev,
             double* tol, double* resid, int* ncv, double* v, int* ldv,
             int* iparam, int* ipntr, double* workd, double* workl, int* lworkl,
             int* info);
}

// The reason we are not using dgesv_ directly is because we have to solve Mx=b
// multiple times where M is fixed but b keeps on changing. The advantage of
// using dgetrf_ and dgetrs_ independently is that we only need to do the LU
// decomposition of M once and then store it for future use, which is all that
// this class does.
class linear_solver_LU {
 public:
  linear_solver_LU(Matrix& matrix_a)
      : mLhsMatrix(matrix_a),
        mNumRows(static_cast<int>(matrix_a.rows())),
        mPivotInfo(matrix_a.rows(), 0.0) {}

  void solve(DenseVector<double>& result) {
    int matrix_a_spacing = mLhsMatrix.spacing();
    int info;

    if (not mLuFactorizationAlreadyDone) {
      dgetrf_(&mNumRows, &mNumRows, mLhsMatrix.data(), &matrix_a_spacing,
              mPivotInfo.data(), &info);

      if (UNLIKELY(info != 0)) {
        ERROR(
            "Lapack failed to compute the LU decomposition. Lapack's dgetrf "
            "INFO = "
            << info);
      }

      mLuFactorizationAlreadyDone = true;
    }

    int result_a_spacing = result.spacing();
    char trans = 'N';
    int nrhs = 1;

    dgetrs_(&trans, &mNumRows, &nrhs, mLhsMatrix.data(), &matrix_a_spacing,
            mPivotInfo.data(), result.data(), &result_a_spacing, &info);

    if (UNLIKELY(info != 0)) {
      ERROR(
          "Lapack failed to solve the linear equation. Lapack's dgetrs "
          "INFO = "
          << info);
    }

    return;
  }

 private:
  Matrix& mLhsMatrix;
  int mNumRows;
  std::vector<int> mPivotInfo;
  bool mLuFactorizationAlreadyDone = false;
};

// This function calls dnaupd and dneupd functions of the Arpack library to
// solve a generalized eigenvalue problem by converting it into a standard
// eigenvalue problem. It uses shift invert transform to make the convergence
// faster and calls Lapack LU solver routine to solve the linear system.
// Documentation for the dnaupd function can be found in the arpack guide page
// 125 or at the link:
// (https://www.caam.rice.edu/software/ARPACK/UG/
// node137.html#SECTION001220000000000000000)
// Some addition information about the choice of algorithm is present on the
// pages 91 and 36.
void arpack_dnaupd_wrapper(linear_solver_LU& M_LUsolver, Matrix& B_input,
                           const double sigma,
                           const std::string which_eigenvalues_to_find,
                           const int number_of_eigenvalues_to_find,
                           DataVector& eigenvalues_real_part,
                           DataVector& eigenvalues_imaginary_part,
                           Matrix& eigenvectors) {
  int num_rows = B_input.rows();
  int nev = number_of_eigenvalues_to_find;

  const char bmat = 'I';
  const std::string howmny = "A";

  // ncv should satisfy ncv <= num_rows and ncv - nev >= 2
  int ncv = 2 * nev;
  int ldv = num_rows;

  int ldz = num_rows + 1;

  int lworkl = 3 * (ncv * ncv) + 6 * ncv;

  double tol = 0.0;       // tol <= 0 means machine precision will be used
  double sigmar = sigma;  // Real part of the sigma
  double sigmai = 0;      // Imaginary part of the sigma
  int rvec = 1;

  DenseVector<double> resid(static_cast<std::size_t>(num_rows), 0.0);
  DenseVector<double> V(static_cast<std::size_t>(ncv * num_rows), 0.0);
  DenseVector<double> workd(3 * static_cast<std::size_t>(num_rows), 0.0);
  DenseVector<double> workl(static_cast<std::size_t>(lworkl), 0.0);
  DenseVector<double> workev(static_cast<std::size_t>(3 * ncv), 0.0);

  // We are declaring vectors of size nev+1 so that there is enough space for
  // Arpack to work with the possible complex conjugate of an eigenpair even
  // if it was not requested by the user.
  DenseVector<double> dr(
      static_cast<std::size_t>((nev + 1)));  // Real Eigenvalues
  DenseVector<double> di(
      static_cast<std::size_t>((nev + 1)));  // Imaginary Eigenvalues
  DenseVector<double> z(
      static_cast<std::size_t>((num_rows + 1) * (nev + 1)));  // Eigenvectors

  std::array<int, 11> iparam{};

  iparam[0] = 1;
  iparam[2] = 10 * num_rows;  // max iterations
  iparam[3] = 1;
  iparam[4] = 0;
  iparam[6] = 3;  // page 91, choses the mode of operation for Arpack

  std::array<int, 14> ipntr{};

  int info = 0, ido = 0;
  while (ido != 99) {
    dnaupd_(&ido, &bmat, &num_rows, which_eigenvalues_to_find.c_str(), &nev,
            &tol, resid.data(), &ncv, V.data(), &ldv, iparam.data(),
            ipntr.data(), workd.data(), workl.data(), &lworkl, &info);

    DenseVector<double> rhs_vec(static_cast<std::size_t>(num_rows), 0.0);
    for (size_t i = 0; i < static_cast<std::size_t>(num_rows); i++) {
      rhs_vec[i] = workd[static_cast<std::size_t>(ipntr[0] - 1) + i];
    }

    DenseVector<double> B_rhs_vec = B_input * rhs_vec;
    M_LUsolver.solve(B_rhs_vec);

    for (size_t i = 0; i < static_cast<std::size_t>(num_rows); i++) {
      workd[static_cast<std::size_t>(ipntr[1] - 1) + i] = B_rhs_vec[i];
    }
  }
  if (UNLIKELY(info != 0)) {
    ERROR(
        "Arpack dnaupd failed. Refer to page 127 of the Arpack manual or "
        "DNAUPD section of https://www.caam.rice.edu/software/ARPACK/UG/ug.html"
        "INFO = "
        << info);
  }

  DenseVector<int> select(static_cast<std::size_t>(ncv), 1);

  dneupd_(&rvec, howmny.c_str(), select.data(), dr.data(), di.data(), z.data(),
          &ldz, &sigmar, &sigmai, workev.data(), &bmat, &num_rows,
          which_eigenvalues_to_find.c_str(), &nev, &tol, resid.data(), &ncv,
          V.data(), &ldv, iparam.data(), ipntr.data(), workd.data(),
          workl.data(), &lworkl, &info);

  if (UNLIKELY(info != 0)) {
    ERROR(
        "Arpack dneupd failed. Refer to page 127 of the Arpack manual or "
        "DNAUPD section of "
        "https://www.caam.rice.edu/software/ARPACK/UG/ug.html."
        "Error codes mean same thing as in dnaupd."
        "INFO = "
        << info);
  }

  for (size_t i = 0; i < static_cast<size_t>(number_of_eigenvalues_to_find);
       i++) {
    eigenvalues_real_part.data()[i] = dr[i];
    eigenvalues_imaginary_part.data()[i] = di[i];
    for (size_t j = 0; j < static_cast<size_t>(num_rows); j++) {
      eigenvectors(j, i) = z[i * static_cast<size_t>(num_rows + 1) + j];
    }
  }

  // The following comments are only relevant if one is interested in using this
  // function to solve eigenvalue problems with complex eigenvectors. Because,
  // in this particular case we know in advance that the eigenvectors are going
  // to be real they are here only for future reference.

  // If the eigenvalues are imaginary, then the real and imaginary parts of the
  // corresponding eigenvector are stored consecutively in z( real part at
  // 'i' and complex part at 'i+1'). The next eigenvector and eigenvalue
  // will be the complex conjugate of these.

  // NOTE: on the page 36 of the Arpack manual it is says that "When the
  // eigenvectors corresponding to a complex comjugate pair of eigenvalues are
  // computed, the vector corresponding to the eigenvalue with positive
  // imaginary part is stored with real and imaginary parts in consecutive
  // columns ...". In my limited testing this does not seem to hold true when
  // shift invert tranforms are used. Even the scipy code seems to follow the
  // procedure described in the last comment:
  // (https://github.com/scipy/scipy/blob
  // /f397c8e17ac235316da6f1a10692b2604b6d6ec4/scipy/sparse/linalg/eigen/arpack
  // /arpack.py#L801)

  // Note that 2 rows of the eigenvector matrix are required to store both the
  // imaginary and the real part of a complex eigenvector. It is responsibility
  // of the user to ensure that the variable `number_of_eigenvalues_to_find` is
  // large enough to accommodate all the eigenvectors of interest.

  return;
}

void find_generalized_eigenvalues_arpack(
    DataVector& eigenvalues_real_part, DataVector& eigenvalues_imaginary_part,
    Matrix& eigenvectors, Matrix& matrix_a, Matrix& matrix_b,
    const size_t number_of_eigenvalues_to_find, const double sigma,
    const std::string which_eigenvalues_to_find) noexcept {
  const size_t number_of_rows = matrix_a.rows();
  ASSERT(number_of_rows == matrix_a.columns(),
         "Matrix A should be square, but A has "
             << matrix_a.rows() << " rows and " << matrix_a.columns()
             << " columns.");
  ASSERT(number_of_rows == matrix_b.rows() and
             number_of_rows == matrix_b.columns(),
         "Matrix A and matrix B should be the same size, but A has "
             << matrix_a.rows() << " rows and " << matrix_a.columns()
             << " columns, while B has " << matrix_b.rows() << " rows and "
             << matrix_b.columns() << " columns.");
  ASSERT(
      number_of_eigenvalues_to_find == eigenvalues_real_part.size() and
          number_of_eigenvalues_to_find == eigenvalues_imaginary_part.size() and
          number_of_eigenvalues_to_find == eigenvectors.columns(),
      "Eigenvalue DataVector size and number of columns in the "
      "eigenvector Matrix should be equal to the number of "
      "eigenvalues to be found, which has the value "
          << number_of_eigenvalues_to_find
          << ", while the real eigenvalues DataVector size is "
          << eigenvalues_real_part.size()
          << " ,the imaginary eigenvalues DataVector size is "
          << eigenvalues_imaginary_part.size()
          << " and the number of columns in the eigenvector Matrix are "
          << eigenvectors.columns() << ".");
  ASSERT(number_of_rows == eigenvectors.rows(),
         "Matrix A and matrix eigenvectors should have the same number of "
         "rows, but A has "
             << matrix_a.rows() << " rows, while the eigenvectors matrix "
             << "has " << eigenvectors.rows() << " rows.");

  ASSERT(which_eigenvalues_to_find.compare("LM") == 0 ||
             which_eigenvalues_to_find.compare("SM") == 0 ||
             which_eigenvalues_to_find.compare("LA") == 0 ||
             which_eigenvalues_to_find.compare("SA") == 0 ||
             which_eigenvalues_to_find.compare("BE") == 0,
         "which_eigenvalues_to_find must be one of \"LM\", "
         "\"SM\", \"LA\", "
         "\"SA\", \"BE\" but it is "
             << which_eigenvalues_to_find << " .");

  matrix_a = matrix_a - sigma * matrix_b;

  linear_solver_LU matrix_M_LU_solver(matrix_a);

  arpack_dnaupd_wrapper(matrix_M_LU_solver, matrix_b, sigma,
                        which_eigenvalues_to_find,
                        number_of_eigenvalues_to_find, eigenvalues_real_part,
                        eigenvalues_imaginary_part, eigenvectors);

  return;
}
