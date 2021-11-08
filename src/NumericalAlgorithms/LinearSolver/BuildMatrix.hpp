// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <tuple>

#include "DataStructures/DenseMatrix.hpp"
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
 * We currently store the matrix representation in a `DenseMatrix` because Blaze
 * doesn't support the inversion of sparse matrices (yet).
 *
 * \param matrix Output buffer for the operator matrix. Must be sized correctly
 * on entry.
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
          typename... OperatorArgs>
void build_matrix(
    const gsl::not_null<DenseMatrix<double, blaze::columnMajor>*> matrix,
    const gsl::not_null<OperandType*> operand_buffer,
    const gsl::not_null<ResultType*> result_buffer,
    const LinearOperator& linear_operator,
    const std::tuple<OperatorArgs...>& operator_args = {}) {
  size_t i = 0;
  // Re-using the iterators for all operator invocations
  auto result_iterator_begin = result_buffer->begin();
  auto result_iterator_end = result_buffer->end();
  for (double& unit_vector_data : *operand_buffer) {
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
    std::copy(result_iterator_begin, result_iterator_end,
              column(*matrix, i).begin());
    ++i;
  }
}

}  // namespace LinearSolver::Serial
