// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function for taking the determinant of a rank-2 tensor

#pragma once

#include "DataStructures/Tensor/Tensor.hpp"

namespace detail {
template <typename Symm, typename Index, typename = std::nullptr_t>
struct DeterminantImpl;

template <typename Symm, typename Index>
struct DeterminantImpl<Symm, Index, Requires<Index::dim == 1>> {
  template <typename T>
  static typename T::type apply(const T& tensor) {
    return get<0, 0>(tensor);
  }
};

template <typename Symm, typename Index>
struct DeterminantImpl<Symm, Index, Requires<Index::dim == 2>> {
  template <typename T>
  static typename T::type apply(const T& tensor) {
    const auto& t00 = get<0, 0>(tensor);
    const auto& t01 = get<0, 1>(tensor);
    const auto& t10 = get<1, 0>(tensor);
    const auto& t11 = get<1, 1>(tensor);
    return t00 * t11 - t01 * t10;
  }
};

template <typename Index>
struct DeterminantImpl<Symmetry<2, 1>, Index, Requires<Index::dim == 3>> {
  template <typename T>
  static typename T::type apply(const T& tensor) {
    const auto& t00 = get<0, 0>(tensor);
    const auto& t01 = get<0, 1>(tensor);
    const auto& t02 = get<0, 2>(tensor);
    const auto& t10 = get<1, 0>(tensor);
    const auto& t11 = get<1, 1>(tensor);
    const auto& t12 = get<1, 2>(tensor);
    const auto& t20 = get<2, 0>(tensor);
    const auto& t21 = get<2, 1>(tensor);
    const auto& t22 = get<2, 2>(tensor);
    return t00 * (t11 * t22 - t12 * t21) - t01 * (t10 * t22 - t12 * t20) +
           t02 * (t10 * t21 - t11 * t20);
  }
};

template <typename Index>
struct DeterminantImpl<Symmetry<1, 1>, Index, Requires<Index::dim == 3>> {
  template <typename T>
  static typename T::type apply(const T& tensor) {
    const auto& t00 = get<0, 0>(tensor);
    const auto& t01 = get<0, 1>(tensor);
    const auto& t02 = get<0, 2>(tensor);
    const auto& t11 = get<1, 1>(tensor);
    const auto& t12 = get<1, 2>(tensor);
    const auto& t22 = get<2, 2>(tensor);
    return t00 * (t11 * t22 - t12 * t12) - t01 * (t01 * t22 - t12 * t02) +
           t02 * (t01 * t12 - t11 * t02);
  }
};

template <typename Index>
struct DeterminantImpl<Symmetry<2, 1>, Index, Requires<Index::dim == 4>> {
  template <typename T>
  static typename T::type apply(const T& tensor) {
    const auto& t00 = get<0, 0>(tensor);
    const auto& t01 = get<0, 1>(tensor);
    const auto& t02 = get<0, 2>(tensor);
    const auto& t03 = get<0, 3>(tensor);
    const auto& t10 = get<1, 0>(tensor);
    const auto& t11 = get<1, 1>(tensor);
    const auto& t12 = get<1, 2>(tensor);
    const auto& t13 = get<1, 3>(tensor);
    const auto& t20 = get<2, 0>(tensor);
    const auto& t21 = get<2, 1>(tensor);
    const auto& t22 = get<2, 2>(tensor);
    const auto& t23 = get<2, 3>(tensor);
    const auto& t30 = get<3, 0>(tensor);
    const auto& t31 = get<3, 1>(tensor);
    const auto& t32 = get<3, 2>(tensor);
    const auto& t33 = get<3, 3>(tensor);
    const auto minor1 = t22 * t33 - t23 * t32;
    const auto minor2 = t21 * t33 - t23 * t31;
    const auto minor3 = t20 * t33 - t23 * t30;
    const auto minor4 = t21 * t32 - t22 * t31;
    const auto minor5 = t20 * t32 - t22 * t30;
    const auto minor6 = t20 * t31 - t21 * t30;
    return t00 * (t11 * minor1 - t12 * minor2 + t13 * minor4) -
           t01 * (t10 * minor1 - t12 * minor3 + t13 * minor5) +
           t02 * (t10 * minor2 - t11 * minor3 + t13 * minor6) -
           t03 * (t10 * minor4 - t11 * minor5 + t12 * minor6);
  }
};

template <typename Index>
struct DeterminantImpl<Symmetry<1, 1>, Index, Requires<Index::dim == 4>> {
  template <typename T>
  static typename T::type apply(const T& tensor) {
    const auto& t00 = get<0, 0>(tensor);
    const auto& t01 = get<0, 1>(tensor);
    const auto& t02 = get<0, 2>(tensor);
    const auto& t03 = get<0, 3>(tensor);
    const auto& t11 = get<1, 1>(tensor);
    const auto& t12 = get<1, 2>(tensor);
    const auto& t13 = get<1, 3>(tensor);
    const auto& t22 = get<2, 2>(tensor);
    const auto& t23 = get<2, 3>(tensor);
    const auto& t33 = get<3, 3>(tensor);
    const auto minor1 = t22 * t33 - t23 * t23;
    const auto minor2 = t12 * t33 - t23 * t13;
    const auto minor3 = t02 * t33 - t23 * t03;
    const auto minor4 = t12 * t23 - t22 * t13;
    const auto minor5 = t02 * t23 - t22 * t03;
    const auto minor6 = t02 * t13 - t12 * t03;
    return t00 * (t11 * minor1 - t12 * minor2 + t13 * minor4) -
           t01 * (t01 * minor1 - t12 * minor3 + t13 * minor5) +
           t02 * (t01 * minor2 - t11 * minor3 + t13 * minor6) -
           t03 * (t01 * minor4 - t11 * minor5 + t12 * minor6);
  }
};
}  // namespace detail

/*!
 * \ingroup TensorGroup
 * \brief Computes the determinant of a rank-2 Tensor `tensor`.
 *
 * \returns The determinant of `tensor`.
 * \requires That `tensor` be a rank-2 Tensor, with both indices sharing the
 *           same dimension and type.
 */
template <typename T, typename Symm, typename Index0, typename Index1>
Scalar<T> determinant(
    const Tensor<T, Symm, index_list<Index0, Index1>>& tensor) {
  static_assert(Index0::dim == Index1::dim,
                "Cannot take the determinant of a Tensor whose Indices are not "
                "of the same dimensionality.");
  static_assert(Index0::index_type == Index1::index_type,
                "Taking the determinant of a mixed Spatial and Spacetime index "
                "Tensor is not allowed since it's not clear what that means.");
  return Scalar<T>{detail::DeterminantImpl<Symm, Index0>::apply(tensor)};
}
