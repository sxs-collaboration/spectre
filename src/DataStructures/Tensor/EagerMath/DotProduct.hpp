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
 * \brief Compute the Euclidean dot product of two vectors or one forms
 *
 * \details
 * Returns \f$A^a B^b \delta_{ab}\f$ for input vectors \f$A^a\f$ and \f$B^b\f$
 * or \f$A_a B_b \delta^{ab}\f$ for input one forms \f$A_a\f$ and \f$B_b\f$.
 */
template <typename DataType, typename Index>
Scalar<DataType> dot_product(
    const Tensor<DataType, Symmetry<1>, index_list<Index>>& vector_a,
    const Tensor<DataType, Symmetry<1>, index_list<Index>>& vector_b) noexcept {
  auto dot_product = make_with_value<Scalar<DataType>>(vector_a, 0.);
  for (size_t d = 0; d < Index::dim; ++d) {
    get(dot_product) += vector_a.get(d) * vector_b.get(d);
  }
  return dot_product;
}

/*!
 * \ingroup TensorGroup
 * \brief Compute the dot product of a vector and a one form
 *
 * \details
 * Returns \f$A^a B_b \delta_{a}^b\f$ for input vector \f$A^a\f$ and
 * input one form \f$B_b\f$
 * or \f$A_a B^b \delta^a_b\f$ for input one form \f$A_a\f$ and
 * input vector \f$B^b\f$.
 */
template <typename DataType, typename Index>
Scalar<DataType> dot_product(
    const Tensor<DataType, Symmetry<1>, index_list<Index>>& vector_a,
    const Tensor<DataType, Symmetry<1>, index_list<change_index_up_lo<Index>>>&
        vector_b) noexcept {
  auto dot_product = make_with_value<Scalar<DataType>>(vector_a, 0.);
  for (size_t d = 0; d < Index::dim; ++d) {
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
    const Tensor<DataType, Symmetry<1>, index_list<Index>>& vector_a,
    const Tensor<DataType, Symmetry<1>, index_list<Index>>& vector_b,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index>,
                            change_index_up_lo<Index>>>& metric) noexcept {
  auto dot_product = make_with_value<Scalar<DataType>>(vector_a, 0.);
  for (size_t a = 0; a < Index::dim; ++a) {
    for (size_t b = 0; b < Index::dim; ++b) {
      get(dot_product) += vector_a.get(a) * vector_b.get(b) * metric.get(a, b);
    }
  }
  return dot_product;
}
