// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearAlgebra/FindRealEigensystem.hpp"

#include <array>
#include <blaze/math/lapack/clapack/gesv.h>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

// LAPACK routines to balance a matrix and to compute and reorder its real Schur
// form. The linear solve uses Blaze's wrapper of dgesv.
extern "C" {
// The final arguments are the "hidden" lengths of the character arguments.
// https://gcc.gnu.org/onlinedocs/gfortran/Argument-passing-conventions.html
extern void dgebal_(char*, int*, double*, int*, int*, int*, double*, int*,
                    size_t);
extern void dgees_(char*, char*, int (*)(double*, double*), int*, double*, int*,
                   int*, double*, double*, double*, int*, double*, int*, int*,
                   int*, size_t, size_t);
extern void dtrsen_(char*, char*, int*, int*, double*, int*, double*, int*,
                    double*, double*, int*, double*, double*, double*, int*,
                    int*, int*, int*, size_t, size_t);
}

template <size_t MatrixSize>
void find_real_eigensystem(
    const gsl::not_null<tnsr::i<DataVector, MatrixSize>*> eigenvalues,
    const gsl::not_null<tnsr::ij<DataVector, MatrixSize>*> right_eigenvectors,
    const gsl::not_null<tnsr::IJ<DataVector, MatrixSize>*> left_eigenvectors,
    const tnsr::iJ<DataVector, MatrixSize>& matrix) {
  // Make sure the outputs have the right number of grid points.
  const size_t num_points = get<0, 0>(matrix).size();
  set_number_of_grid_points(eigenvalues, num_points);
  set_number_of_grid_points(right_eigenvectors, num_points);
  set_number_of_grid_points(left_eigenvectors, num_points);

  // Matrix size, shared by all LAPACK calls below.
  // Note: LAPACK takes column-major arrays, and its options and sizes are not
  //       const because LAPACK does not want them const.
  auto lapack_matrix_size = static_cast<int>(MatrixSize);

  // LAPACK uses an integer called info to return its status.
  // info = 0 : success
  // info = -i: ith argument had bad value
  // info > 0 : the computation failed (dgees, dtrsen and dgesv only)
  int info = 0;

  // Loop over each grid point.
  for (size_t point = 0; point < num_points; ++point) {
    // Build matrix at this grid point.
    // Note: LAPACK overwrites this matrix in place. It holds A, then the
    //       balanced A' (dgebal), then the Schur form T (dgees).
    // Note: subnormal entries (below ~2.2e-308) are flushed to zero because
    //       balancing would otherwise give scale factors up to ~1e161 that
    //       might leave the right eigenvectors numerically singular.
    std::array<double, MatrixSize * MatrixSize> lapack_matrix{};
    for (size_t row = 0; row < MatrixSize; ++row) {
      for (size_t col = 0; col < MatrixSize; ++col) {
        const double entry = matrix.get(row, col)[point];
        // Copy only normal entries; subnormal ones keep the initial zero.
        if (std::abs(entry) >= std::numeric_limits<double>::min()) {
          gsl::at(lapack_matrix, row + (MatrixSize * col)) = entry;
        }
      }
    }

    // Balance the matrix by scaling, A -> A' = D^{-1} A D.
    // 'S' means only scale rows and columns (no permutation), so D is diagonal.
    char balance_by_scaling = 'S';
    // The scale factors are the diagonal entries of D.
    std::array<double, MatrixSize> scale_factors{};
    // dgebal returns the first and last indices of the rows and columns it
    // balanced. When only scaling, these are always 1 and MatrixSize.
    int first_balanced_index = 0;
    int last_balanced_index = 0;
    dgebal_(&balance_by_scaling, &lapack_matrix_size, lapack_matrix.data(),
            &lapack_matrix_size, &first_balanced_index, &last_balanced_index,
            scale_factors.data(), &info, 1);
    ASSERT(info == 0, "dgebal failed with info = " << info << " at grid point "
                                                   << point << ".");

    // Compute the real Schur form of the balanced matrix, A' = Z T Z^T.
    // 'V' means compute the Schur vectors, i.e., the columns of Z.
    char compute_schur_vectors = 'V';
    // 'N' means do not sort the eigenvalues on the diagonal of T.
    char do_not_sort = 'N';
    std::array<double, MatrixSize * MatrixSize> schur_vectors{};
    std::array<double, MatrixSize> eigenvalues_real_part{};
    std::array<double, MatrixSize> eigenvalues_imaginary_part{};
    // dgees returns the number of sorted eigenvalues, which is 0 when not
    // sorting. Its bool work array is also only used when sorting.
    int number_of_sorted_eigenvalues = 0;
    std::array<int, MatrixSize> schur_bool_work{};
    // The work array has LAPACK's minimum size, 3 * MatrixSize. A larger one
    // only speeds up large matrices, but we only expect to use
    // MatrixSize <~ 10 in the near term.
    std::array<double, 3 * MatrixSize> schur_work{};
    auto schur_work_size = static_cast<int>(schur_work.size());
    dgees_(&compute_schur_vectors, &do_not_sort, nullptr, &lapack_matrix_size,
           lapack_matrix.data(), &lapack_matrix_size,
           &number_of_sorted_eigenvalues, eigenvalues_real_part.data(),
           eigenvalues_imaginary_part.data(), schur_vectors.data(),
           &lapack_matrix_size, schur_work.data(), &schur_work_size,
           schur_bool_work.data(), &info, 1, 1);
    if (UNLIKELY(info != 0)) {
      ERROR("dgees failed with info = " << info << " at grid point " << point
                                        << ".");
    }
    // Get the tolerance for comparing eigenvalues.
    // Note: the tolerance is relative to the Frobenius norm of the balanced
    //       matrix A'. dgees has overwritten A' with T, which has the same
    //       Frobenius norm because Z is orthogonal.
    constexpr double relative_tolerance = 1.0e-10;
    double frobenius_norm_squared = 0.0;
    for (const double entry : lapack_matrix) {
      frobenius_norm_squared += square(entry);
    }
    const double tolerance =
        relative_tolerance * std::sqrt(frobenius_norm_squared);

    // Check that all eigenvalues are real, as this function assumes.
    // Note: round-off can split a degenerate eigenvalue into a complex pair
    //       with a tiny imaginary part. Such a pair has a single real part,
    //       so it is grouped into one family below.
    for (size_t i = 0; i < MatrixSize; ++i) {
      ASSERT(std::abs(gsl::at(eigenvalues_imaginary_part, i)) <= tolerance,
             "Complex eigenvalue: " << gsl::at(eigenvalues_real_part, i)
                                    << " + "
                                    << gsl::at(eigenvalues_imaginary_part, i)
                                    << " i at grid point " << point << ".");
    }

    // Group the eigenvalues into degenerate families.
    // Note: sorted by real part, a new family starts where the gap between
    //       neighbors exceeds the tolerance.
    // Sort the eigenvalue indices by real part.
    std::array<size_t, MatrixSize> sorted_eigenvalue_indices{};
    alg::iota(sorted_eigenvalue_indices, size_t{0});
    alg::sort(sorted_eigenvalue_indices,
              [&eigenvalues_real_part](const size_t i, const size_t j) {
                return gsl::at(eigenvalues_real_part, i) <
                       gsl::at(eigenvalues_real_part, j);
              });
    // Label each eigenvalue with its family, counting families from 0.
    std::array<size_t, MatrixSize> eigenvalue_families{};
    size_t number_of_families = 1;
    gsl::at(eigenvalue_families, gsl::at(sorted_eigenvalue_indices, 0)) = 0;
    for (size_t i = 1; i < MatrixSize; ++i) {
      const size_t eigenvalue_index = gsl::at(sorted_eigenvalue_indices, i);
      const size_t previous_index = gsl::at(sorted_eigenvalue_indices, i - 1);
      if (gsl::at(eigenvalues_real_part, eigenvalue_index) -
              gsl::at(eigenvalues_real_part, previous_index) >
          tolerance) {
        ++number_of_families;
      }
      gsl::at(eigenvalue_families, eigenvalue_index) = number_of_families - 1;
    }

    // Get the right eigenvectors family by family.
    // Note: reordering the Schur form so that a family's eigenvalues come
    //       first makes the leading Schur vectors an orthonormal basis of the
    //       family's invariant subspace. Each vector gets its own eigenvalue
    //       from the reordered Schur form.
    std::array<double, MatrixSize * MatrixSize> balanced_right_eigenvectors{};
    size_t eigenvector_index = 0;
    for (size_t family = 0; family < number_of_families; ++family) {
      // Select the eigenvalues of this family.
      // Note: dtrsen takes the selection as a Fortran logical array, so 1
      //       means in the family and 0 means not.
      std::array<int, MatrixSize> selected_eigenvalues{};
      size_t family_size = 0;
      for (size_t i = 0; i < MatrixSize; ++i) {
        const bool is_in_family = gsl::at(eigenvalue_families, i) == family;
        gsl::at(selected_eigenvalues, i) = is_in_family ? 1 : 0;
        family_size += is_in_family ? 1 : 0;
      }
      // Check whether the family already comes first in T, in which case it
      // needs no reordering.
      bool family_is_leading = true;
      for (size_t i = 0; i < family_size; ++i) {
        if (gsl::at(eigenvalue_families, i) != family) {
          family_is_leading = false;
        }
      }
      // Copy T, Z and the eigenvalues, which dtrsen overwrites, so that every
      // family is reordered from the same Schur form.
      // Note: reordering from the previous family's result would also work,
      //       but the family labels refer to positions in the original T, so
      //       we would have to track the permutation dtrsen applies.
      std::array<double, MatrixSize * MatrixSize> family_schur_form =
          lapack_matrix;
      std::array<double, MatrixSize * MatrixSize> family_schur_vectors =
          schur_vectors;
      std::array<double, MatrixSize> family_eigenvalues_real_part =
          eigenvalues_real_part;
      if (not family_is_leading) {
        // Reorder the Schur form so that the family comes first,
        // T -> Q^T T Q and Z -> Z Q.
        // 'N' means do not compute condition numbers.
        char no_condition_numbers = 'N';
        // 'V' means also update the Schur vectors, Z -> Z Q.
        char update_schur_vectors = 'V';
        // dtrsen returns the reordered eigenvalues. Their imaginary parts are
        // not used, since all eigenvalues are real.
        std::array<double, MatrixSize> family_eigenvalues_imaginary_part{};
        // dtrsen returns the number of selected eigenvalues, which should be
        // the family size. It also returns the condition numbers of the
        // family's eigenvalues and invariant subspace, which are not computed
        // with 'N'.
        int reordered_family_size = 0;
        double eigenvalues_condition_number = 0.0;
        double invariant_subspace_condition_number = 0.0;
        // The work arrays have LAPACK's minimum sizes without condition
        // numbers, MatrixSize doubles and one int.
        std::array<double, MatrixSize> reorder_work{};
        auto reorder_work_size = static_cast<int>(reorder_work.size());
        int reorder_int_work = 0;
        int reorder_int_work_size = 1;
        dtrsen_(&no_condition_numbers, &update_schur_vectors,
                selected_eigenvalues.data(), &lapack_matrix_size,
                family_schur_form.data(), &lapack_matrix_size,
                family_schur_vectors.data(), &lapack_matrix_size,
                family_eigenvalues_real_part.data(),
                family_eigenvalues_imaginary_part.data(),
                &reordered_family_size, &eigenvalues_condition_number,
                &invariant_subspace_condition_number, reorder_work.data(),
                &reorder_work_size, &reorder_int_work, &reorder_int_work_size,
                &info, 1, 1);
        if (UNLIKELY(info != 0 or
                     std::cmp_not_equal(reordered_family_size, family_size))) {
          ERROR("dtrsen failed with info = "
                << info << " at grid point " << point << " (family of "
                << family_size << " eigenvalues, " << reordered_family_size
                << " reordered).");
        }
      }
      // Store the family's eigenvalues and leading Schur vectors.
      for (size_t i = 0; i < family_size; ++i) {
        eigenvalues->get(eigenvector_index)[point] =
            gsl::at(family_eigenvalues_real_part, i);
        for (size_t k = 0; k < MatrixSize; ++k) {
          gsl::at(balanced_right_eigenvectors,
                  k + (MatrixSize * eigenvector_index)) =
              gsl::at(family_schur_vectors, k + (MatrixSize * i));
        }
        ++eigenvector_index;
      }
    }

    // Get the left eigenvectors as the inverse of the right ones, L' = R'^{-1}.
    // Note: dgesv solves for X in the linear system M X = B. With M = R' and
    //       B = I, the solution is X = R'^{-1} = L'.
    // Note: we use dgesv instead of blaze::invert, which is less accurate when
    //       R' is close to singular.
    // dgesv overwrites the system matrix M, so it gets a copy of R'.
    std::array<double, MatrixSize * MatrixSize> system_matrix =
        balanced_right_eigenvectors;
    // dgesv overwrites B with the solution X, so B starts as the identity and
    // ends as L'.
    std::array<double, MatrixSize * MatrixSize> balanced_left_eigenvectors{};
    for (size_t i = 0; i < MatrixSize; ++i) {
      gsl::at(balanced_left_eigenvectors, i + (MatrixSize * i)) = 1.0;
    }
    // B has MatrixSize columns, so dgesv solves one system per column.
    const int number_of_right_hand_sides = lapack_matrix_size;
    // dgesv also returns pivots, which we do not use.
    std::array<int, MatrixSize> pivots{};
    blaze::gesv(lapack_matrix_size, number_of_right_hand_sides,
                system_matrix.data(), lapack_matrix_size, pivots.data(),
                balanced_left_eigenvectors.data(), lapack_matrix_size, &info);
    if (UNLIKELY(info != 0)) {
      ERROR("dgesv failed with info = " << info << " at grid point " << point
                                        << ".");
    }

    // Undo the balancing: R = D R' and L = L' D^{-1}.
    // Note: we're looping through the ith eigenvalue/vectors, not the ith row.
    for (size_t i = 0; i < MatrixSize; ++i) {
      for (size_t k = 0; k < MatrixSize; ++k) {
        right_eigenvectors->get(i, k)[point] =
            gsl::at(balanced_right_eigenvectors, k + (MatrixSize * i)) *
            gsl::at(scale_factors, k);
        left_eigenvectors->get(i, k)[point] =
            gsl::at(balanced_left_eigenvectors, i + (MatrixSize * k)) /
            gsl::at(scale_factors, k);
      }
    }
  }
}

#define GET_SIZE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void find_real_eigensystem<GET_SIZE(data)>(                         \
      gsl::not_null<tnsr::i<DataVector, GET_SIZE(data)>*> eigenvalues,         \
      gsl::not_null<tnsr::ij<DataVector, GET_SIZE(data)>*> right_eigenvectors, \
      gsl::not_null<tnsr::IJ<DataVector, GET_SIZE(data)>*> left_eigenvectors,  \
      const tnsr::iJ<DataVector, GET_SIZE(data)>& matrix);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

#undef INSTANTIATE
#undef GET_SIZE
