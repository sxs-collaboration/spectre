// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Matrix.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
/// \endcond

/// LAPACK wrappers
namespace lapack {
/// @{
/*!
 * \ingroup LinearSolverGroup
 * \brief Wrapper for LAPACK dgesv, which solves the general linear equation
 * \f$A x = b\f$ by LUP (Lower-triangular, upper-triangular, and permutation)
 * decomposition.
 * \details Several interfaces are provided with combinations of
 * input parameters due to the versatility of the LAPACK utility.
 * - `rhs_in_solution_out` or `solution` : If the `rhs` `DataVector` is not
 * separately supplied, this single pass-by-pointer `DataVector` acts as the
 * input \f$ b\f$, and is overwritten with the solution \f$x\f$; in this case
 * `rhs_in_solution_out` must be sufficiently large to contain the solution. If
 * `rhs` is separately supplied, `solution` will contain \f$x\f$ at the end of
 * the algorithm, and must be sufficiently large to contain the solution.
 * - `rhs` (optional) : An input `DataVector` representing \f$b\f$ that is not
 * overwritten.
 * - `matrix_operator`: The `Matrix` \f$A\f$. If passed by pointer, this matrix
 * will be 'destroyed' and overwritten with LU decomposition information.
 * Optionally, a `pivots` vector may be passed by pointer so that the full LUP
 * decomposition information can be recovered from the LAPACK output. If passed
 * by const reference, the wrapper will make a copy so that LAPACK does not
 * overwrite the supplied matrix.
 * - `pivots` (optional) : a `std::vector<int>`, passed by pointer, which will
 * contain the permutation information necessary to reassemble the full matrix
 * information if LAPACK is permitted to modify the input matrix in-place.
 * - `number_of_rhs` : The number of columns of \f$x\f$ and \f$b\f$. This is
 * thought of as the 'number of right-hand-sides' to be solved for. In most
 * cases, that additional dimension can be inferred from the dimensionality of
 * the matrix and the input vector(s), which occurs if `number_of_rhs` is set to
 * 0 (default). If the provided `DataVector` is an inappropriate size for that
 * inference (i.e. is not a multiple of the largest dimension of the matrix), or
 * if you only want to solve for the first `number_of_rhs` columns, the
 * parameter `number_of_rhs` must be supplied.
 *
 * The function return `int` is the value provided by the `INFO` field of the
 * LAPACK call. It is 0 for a successful linear solve, and nonzero values code
 * for types of failures of the algorithm.
 * See LAPACK documentation for further details about `dgesv`:
 * http://www.netlib.org/lapack/
 */
int general_matrix_linear_solve(gsl::not_null<DataVector*> rhs_in_solution_out,
                                gsl::not_null<Matrix*> matrix_operator,
                                int number_of_rhs = 0) noexcept;

int general_matrix_linear_solve(gsl::not_null<DataVector*> rhs_in_solution_out,
                                gsl::not_null<std::vector<int>*> pivots,
                                gsl::not_null<Matrix*> matrix_operator,
                                int number_of_rhs = 0) noexcept;

int general_matrix_linear_solve(gsl::not_null<DataVector*> solution,
                                gsl::not_null<Matrix*> matrix_operator,
                                const DataVector& rhs,
                                int number_of_rhs = 0) noexcept;

int general_matrix_linear_solve(gsl::not_null<DataVector*> solution,
                                gsl::not_null<std::vector<int>*> pivots,
                                gsl::not_null<Matrix*> matrix_operator,
                                const DataVector& rhs,
                                int number_of_rhs = 0) noexcept;

int general_matrix_linear_solve(gsl::not_null<DataVector*> rhs_in_solution_out,
                                const Matrix& matrix_operator,
                                int number_of_rhs = 0) noexcept;

int general_matrix_linear_solve(gsl::not_null<DataVector*> solution,
                                const Matrix& matrix_operator,
                                const DataVector& rhs,
                                int number_of_rhs = 0) noexcept;
/// @}
}  // namespace lapack
