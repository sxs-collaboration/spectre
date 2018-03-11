// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions euclidean dot_product and dot_product with a metric

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/MakeWithValue.hpp"

/*!
 * \ingroup TensorGroup
 * \brief Contract the indices of a pair of rank-1 tensors
 */
template <typename DataType, typename IndexA, typename IndexB>
Scalar<DataType> dot_product(
    const Tensor<DataType, Symmetry<1>, tmpl::list<IndexA>>& vector_a,
    const Tensor<DataType, Symmetry<1>, tmpl::list<IndexB>>&
        vector_b) noexcept {
  static_assert(can_contract_v<IndexA, IndexB>,
                "Noncontractible tensors passed to dot_product");
  auto dot_product = make_with_value<Scalar<DataType>>(vector_a, 0.);
  for (size_t d = 0; d < IndexA::dim; ++d) {
    get(dot_product) += vector_a.get(d) * vector_b.get(d);
  }
  return dot_product;
}

/*!
 * \ingroup TensorGroup
 * \brief Compute the dot_product of two vectors or one forms
 *
 * \details
 * Returns \f$g_{ab} A^a B^b\f$, where \f$g_{ab}\f$ is the metric,
 * \f$A^a\f$ is vector_a, and \f$B^b\f$ is vector_b.
 * Or, returns \f$g^{ab} A_a B_b\f$ when given one forms \f$A_a\f$
 * and \f$B_b\f$ with an inverse metric \f$g^{ab}\f$.
 */
template <typename DataType, typename Index>
Scalar<DataType> dot_product(
    const Tensor<DataType, Symmetry<1>, tmpl::list<Index>>& vector_a,
    const Tensor<DataType, Symmetry<1>, tmpl::list<Index>>& vector_b,
    const Tensor<DataType, Symmetry<1, 1>,
                 tmpl::list<change_index_up_lo<Index>,
                            change_index_up_lo<Index>>>& metric) noexcept {
  auto dot_product = make_with_value<Scalar<DataType>>(vector_a, 0.);
  for (size_t a = 0; a < Index::dim; ++a) {
    for (size_t b = 0; b < Index::dim; ++b) {
      get(dot_product) += vector_a.get(a) * vector_b.get(b) * metric.get(a, b);
    }
  }
  return dot_product;
}
