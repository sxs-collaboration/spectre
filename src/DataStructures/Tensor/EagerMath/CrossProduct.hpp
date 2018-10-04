// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions `cross_product` for flat and curved-space cross products.

#pragma once

#include <cstddef>

#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/MakeWithValue.hpp"

/*!
 * \ingroup TensorGroup
 * \brief Compute the Euclidean cross product of two vectors or one forms
 *
 * \details
 * Returns \f$A^j B^k \epsilon_{ljk} \delta^{il}\f$ for input vectors \f$A^j\f$
 * and \f$B^k\f$ or \f$A_j B_k \epsilon^{ljk} \delta_{il}\f$ for input one
 * forms \f$A_j\f$ and \f$B_k\f$.
 */
template <typename DataType, typename Index>
Tensor<DataType, Symmetry<1>, index_list<Index>> cross_product(
    const Tensor<DataType, Symmetry<1>, index_list<Index>>& vector_a,
    const Tensor<DataType, Symmetry<1>, index_list<Index>>& vector_b) noexcept {
  static_assert(Index::dim == 3, "cross_product vectors must have dimension 3");
  static_assert(Index::index_type == IndexType::Spatial,
                "cross_product vectors must be spatial");

  auto cross_product =
      make_with_value<Tensor<DataType, Symmetry<1>, index_list<Index>>>(
          vector_a, 0.0);
  for (LeviCivitaIterator<3> it; it; ++it) {
    cross_product.get(it[0]) +=
        it.sign() * vector_a.get(it[1]) * vector_b.get(it[2]);
  }
  return cross_product;
}

/*!
 * \ingroup TensorGroup
 * \brief Compute the Euclidean cross product of a vector and a one form
 *
 * \details
 * Returns \f$A^j B_l \delta^{lk} \epsilon_{ijk}\f$ for input vector \f$A^j\f$
 * and input one form \f$B_l\f$
 * or \f$A_j B^l \delta_{lk} \epsilon^{ijk}\f$ for input one form \f$A_j\f$ and
 * input vector \f$B^l\f$. Note that this function returns a vector if
 * `vector_b` is a vector and a one form if `vector_b` is a one form.
 */
template <typename DataType, typename Index>
Tensor<DataType, Symmetry<1>, index_list<change_index_up_lo<Index>>>
cross_product(
    const Tensor<DataType, Symmetry<1>, index_list<Index>>& vector_a,
    const Tensor<DataType, Symmetry<1>, index_list<change_index_up_lo<Index>>>&
        vector_b) noexcept {
  static_assert(Index::dim == 3, "cross_product vectors must have dimension 3");
  static_assert(Index::index_type == IndexType::Spatial,
                "cross_product vectors must be spatial");

  auto cross_product = make_with_value<
      Tensor<DataType, Symmetry<1>, index_list<change_index_up_lo<Index>>>>(
      vector_b, 0.0);
  for (LeviCivitaIterator<3> it; it; ++it) {
    cross_product.get(it[0]) +=
        it.sign() * vector_a.get(it[1]) * vector_b.get(it[2]);
  }
  return cross_product;
}

/*!
 * \ingroup TensorGroup
 * \brief Compute the cross product of two vectors or one forms
 *
 * \details
 * Returns \f$\sqrt{g} g^{li} A^j B^k \epsilon_{ljk}\f$, where
 * \f$A^j\f$ and \f$B^k\f$ are vectors and \f$g^{li}\f$ and \f$g\f$
 * are the inverse and determinant, respectively, of
 * the spatial metric (computed via `determinant_and_inverse`). In this case,
 * the arguments `vector_a` and `vector_b` should be vectors,
 * the argument `metric_or_inverse_metric` should be the inverse spatial
 * metric  \f$g^{ij}\f$, and the argument `metric_determinant` should be the
 * determinant of the spatial metric \f$\det(g_{ij})\f$.
 * Or, returns \f$\sqrt{g}^{-1} g_{li} A_j B_k \epsilon^{ljk}\f$, where
 * \f$A_j\f$ and \f$B_k\f$ are one forms and \f$g_{li}\f$ and \f$g\f$
 * are the spatial metric and its determinant. In this case,
 * the arguments `vector_a` and `vector_b` should be one forms,
 * the argument `metric_or_inverse_metric` should be the spatial metric
 * \f$g_{ij}\f$, and the argument `metric_determinant` should be the
 * determinant of the spatial metric \f$\det(g_{ij})\f$.
 */
template <typename DataType, typename Index>
Tensor<DataType, Symmetry<1>, index_list<Index>> cross_product(
    const Tensor<DataType, Symmetry<1>, index_list<Index>>& vector_a,
    const Tensor<DataType, Symmetry<1>, index_list<Index>>& vector_b,
    const Tensor<DataType, Symmetry<1, 1>, index_list<Index, Index>>&
        metric_or_inverse_metric,
    const Scalar<DataType>& metric_determinant) noexcept {
  static_assert(Index::dim == 3, "cross_product vectors must have dimension 3");
  static_assert(Index::index_type == IndexType::Spatial,
                "cross_product vectors must be spatial");

  auto cross_product =
      make_with_value<Tensor<DataType, Symmetry<1>, index_list<Index>>>(
          vector_a, 0.);
  for (size_t i = 0; i < Index::dim; ++i) {
    for (LeviCivitaIterator<3> it; it; ++it) {
      cross_product.get(i) += it.sign() * vector_a.get(it[1]) *
                              vector_b.get(it[2]) *
                              metric_or_inverse_metric.get(it[0], i);
    }
    if (Index::ul == UpLo::Up) {
      cross_product.get(i) *= sqrt(get(metric_determinant));
    } else {
      cross_product.get(i) /= sqrt(get(metric_determinant));
    }
  }
  return cross_product;
}

/*!
 * \ingroup TensorGroup
 * \brief Compute the cross product of a vector and a one form
 *
 * \details
 * Returns \f$\sqrt{g} A^j B_l g^{lk} \epsilon_{ijk}\f$ for input vector
 * \f$A^j\f$ and input one form \f$B_l\f$. In this case,
 * the argument `vector_a` should be a vector, `vector_b` should be a one form,
 * `metric_or_inverse_metric` should be the inverse spatial
 * metric  \f$g^{ij}\f$, and `metric_determinant` should be the
 * determinant of the spatial metric \f$\det(g_{ij})\f$.
 * Or, returns \f$\sqrt{g}^{-1} A_j B^l g_{lk}
 * \epsilon^{ijk}\f$ for input one form \f$A_j\f$ and input vector \f$B^l\f$.
 * In this case,
 * the argument `vector_a` should be a one form, `vector_b` should be a vector,
 * `metric_or_inverse_metric` should be the spatial metric
 * \f$g_{ij}\f$, and `metric_determinant` should be the
 * determinant of the spatial metric \f$\det(g_{ij})\f$.
 * Note that this function returns a vector if `vector_b` is a vector and a
 * one form if `vector_b` is a one form.
 */
template <typename DataType, typename Index>
Tensor<DataType, Symmetry<1>, index_list<change_index_up_lo<Index>>>
cross_product(const Tensor<DataType, Symmetry<1>, index_list<Index>>& vector_a,
              const Tensor<DataType, Symmetry<1>,
                           index_list<change_index_up_lo<Index>>>& vector_b,
              const Tensor<DataType, Symmetry<1, 1>, index_list<Index, Index>>&
                  metric_or_inverse_metric,
              const Scalar<DataType>& metric_determinant) noexcept {
  static_assert(Index::dim == 3, "cross_product vectors must have dimension 3");
  static_assert(Index::index_type == IndexType::Spatial,
                "cross_product vectors must be spatial");

  auto cross_product = make_with_value<
      Tensor<DataType, Symmetry<1>, index_list<change_index_up_lo<Index>>>>(
      vector_b, 0.0);
  for (LeviCivitaIterator<3> it; it; ++it) {
    if (Index::ul == UpLo::Up) {
      for (size_t l = 0; l < Index::dim; ++l) {
        cross_product.get(it[0]) += it.sign() * vector_a.get(it[1]) *
                                    vector_b.get(l) *
                                    metric_or_inverse_metric.get(it[2], l);
      }
    } else {
      for (size_t l = 0; l < Index::dim; ++l) {
        cross_product.get(it[0]) += it.sign() * vector_a.get(l) *
                                    vector_b.get(it[2]) *
                                    metric_or_inverse_metric.get(it[1], l);
      }
    }
  }

  for (size_t i = 0; i < Index::dim; ++i) {
    if (Index::ul == UpLo::Up) {
      cross_product.get(i) *= sqrt(get(metric_determinant));
    } else {
      cross_product.get(i) /= sqrt(get(metric_determinant));
    }
  }
  return cross_product;
}
