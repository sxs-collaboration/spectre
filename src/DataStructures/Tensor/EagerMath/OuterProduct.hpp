// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \brief The `Tensor` type resulting from the outer product of two tensors
 *
 * \see outer_product
 */
template <typename DataType, typename SymmLhs, typename IndicesLhs,
          typename SymmRhs, typename IndicesRhs>
using OuterProductResultTensor = Tensor<
    DataType,
    tmpl::append<
        tmpl::transform<
            SymmLhs,
            tmpl::plus<tmpl::_1,
                       tmpl::fold<SymmRhs, tmpl::int32_t<0>,
                                  tmpl::max<tmpl::_state, tmpl::_element>>>>,
        SymmRhs>,
    tmpl::append<IndicesLhs, IndicesRhs>>;

/// @{
/*!
 * \ingroup TensorGroup
 * \brief The outer product (or tensor product) of two tensors
 *
 * \details
 * Computes $A_{i\ldots j\ldots} = B_{i\ldots} C_{j\ldots}$ for two tensors
 * $B_{i\ldots}$ and $C_{j\ldots}$. Both tensors can have arbitrary indices and
 * symmetries.
 */
template <typename DataType, typename SymmLhs, typename IndicesLhs,
          typename SymmRhs, typename IndicesRhs>
void outer_product(const gsl::not_null<OuterProductResultTensor<
                       DataType, SymmLhs, IndicesLhs, SymmRhs, IndicesRhs>*>
                       result,
                   const Tensor<DataType, SymmLhs, IndicesLhs>& lhs,
                   const Tensor<DataType, SymmRhs, IndicesRhs>& rhs) {
  for (auto it_lhs = lhs.begin(); it_lhs != lhs.end(); ++it_lhs) {
    for (auto it_rhs = rhs.begin(); it_rhs != rhs.end(); ++it_rhs) {
      const auto result_indices = concatenate(lhs.get_tensor_index(it_lhs),
                                              rhs.get_tensor_index(it_rhs));
      result->get(result_indices) = *it_lhs * *it_rhs;
    }
  }
}

template <typename DataType, typename SymmLhs, typename IndicesLhs,
          typename SymmRhs, typename IndicesRhs>
auto outer_product(const Tensor<DataType, SymmLhs, IndicesLhs>& lhs,
                   const Tensor<DataType, SymmRhs, IndicesRhs>& rhs)
    -> OuterProductResultTensor<DataType, SymmLhs, IndicesLhs, SymmRhs,
                                IndicesRhs> {
  OuterProductResultTensor<DataType, SymmLhs, IndicesLhs, SymmRhs, IndicesRhs>
      result{};
  ::outer_product(make_not_null(&result), lhs, rhs);
  return result;
}
/// @}
