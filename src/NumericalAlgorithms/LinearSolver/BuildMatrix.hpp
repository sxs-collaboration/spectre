// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <blaze/math/Column.h>
#include <blaze/math/Matrix.h>
#include <blaze/math/typetraits/IsDenseMatrix.h>
#include <blaze/math/typetraits/IsSparseMatrix.h>
#include <cstddef>
#include <tuple>

#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"

namespace LinearSolver::Serial {

namespace detail {
CREATE_IS_CALLABLE(reset)
CREATE_IS_CALLABLE_V(reset)
}  // namespace detail

/*!
 * \brief Construct an explicit matrix representation by "sniffing out" the
 * linear operator, i.e. feeding it unit vectors.
 *
 * \param matrix Output buffer for the operator matrix. Must be sized correctly
 * on entry. Can be any dense or sparse Blaze matrix.
 * \param operand_buffer Memory buffer that can hold operand data for the
 * `linear_operator`. Must be sized correctly on entry, and must be filled with
 * zeros.
 * \param result_buffer Memory buffer that can hold the result of the
 * `linear_operator` applied to the operand. Must be sized correctly on entry.
 * \param linear_operator The linear operator of which the matrix representation
 * should be constructed. See `::LinearSolver::Serial::LinearSolver::solve` for
 * requirements on the linear operator.
 * \param operator_args These arguments are passed along to the
 * `linear_operator` when it is applied to an operand.
 */
template <typename LinearOperator, typename OperandType, typename ResultType,
          typename MatrixType, typename... OperatorArgs>
void build_matrix(const gsl::not_null<MatrixType*> matrix,
                  const gsl::not_null<OperandType*> operand_buffer,
                  const gsl::not_null<ResultType*> result_buffer,
                  const LinearOperator& linear_operator,
                  const std::tuple<OperatorArgs...>& operator_args = {}) {
  static_assert(
      blaze::IsSparseMatrix_v<MatrixType> or blaze::IsDenseMatrix_v<MatrixType>,
      "Unexpected matrix type");
  if constexpr (blaze::IsSparseMatrix_v<MatrixType>) {
    matrix->reset();
  }
  size_t i = 0;
  // Re-using the iterators for all operator invocations
  auto result_iterator_begin = result_buffer->begin();
  auto result_iterator_end = result_buffer->end();
  for (auto& unit_vector_data : *operand_buffer) {
    // Set a 1 at the unit vector location i
    unit_vector_data = 1.;
    // Invoke the operator on the unit vector
    std::apply(
        linear_operator,
        std::tuple_cat(std::forward_as_tuple(result_buffer, *operand_buffer),
                       operator_args));
    // Set the unit vector back to zero
    unit_vector_data = 0.;
    // Reset the iterator by calling its `reset` member function or by
    // re-creating it
    if constexpr (detail::is_reset_callable_v<
                      decltype(result_iterator_begin)>) {
      result_iterator_begin.reset();
    } else {
      result_iterator_begin = result_buffer->begin();
      result_iterator_end = result_buffer->end();
    }
    // Store the result in column i of the matrix
    auto col = column(*matrix, i);
    if constexpr (blaze::IsSparseMatrix_v<MatrixType>) {
      size_t k = 0;
      while (result_iterator_begin != result_iterator_end) {
        if (not equal_within_roundoff(*result_iterator_begin, 0.)) {
          col[k] = *result_iterator_begin;
        }
        ++result_iterator_begin;
        ++k;
      }
    } else {
      std::copy(result_iterator_begin, result_iterator_end, col.begin());
    }
    ++i;
  }
}

}  // namespace LinearSolver::Serial
