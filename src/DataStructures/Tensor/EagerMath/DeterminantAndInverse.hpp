// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function computing the determinant and inverse of a tensor.

#pragma once

#include <type_traits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsComplexOfFundamental.hpp"

namespace determinant_and_inverse_detail {
// Helps to shorten some repeated code:
template <typename Index0, typename Index1>
using inverse_indices =
    tmpl::list<change_index_up_lo<Index1>, change_index_up_lo<Index0>>;

template <typename Symm, typename Index0, typename Index1,
          typename = std::nullptr_t>
struct DetAndInverseImpl;

template <typename Symm, typename Index0, typename Index1>
struct DetAndInverseImpl<Symm, Index0, Index1, Requires<Index0::dim == 1>> {
  template <typename T>
  static void apply(
      const gsl::not_null<Scalar<T>*> det,
      const gsl::not_null<Tensor<T, Symm, inverse_indices<Index0, Index1>>*>
          inv,
      const Tensor<T, Symm, tmpl::list<Index0, Index1>>& tensor) noexcept {
    const T& t00 = get<0, 0>(tensor);
    // inv is non-const so that it can be moved into the std::pair:
    get(*det) = t00;
    get<0, 0>(*inv) = 1.0 / t00;
  }
};

// The inverse of a 2x2 tensor is computed from Cramer's rule.
template <typename Index0, typename Index1>
struct DetAndInverseImpl<Symmetry<2, 1>, Index0, Index1,
                         Requires<Index0::dim == 2>> {
  template <typename T>
  static void apply(
      const gsl::not_null<Scalar<T>*> det,
      const gsl::not_null<
          Tensor<T, Symmetry<2, 1>, inverse_indices<Index0, Index1>>*>
          inv,
      const Tensor<T, Symmetry<2, 1>, tmpl::list<Index0, Index1>>&
          tensor) noexcept {
    const T& t00 = get<0, 0>(tensor);
    const T& t01 = get<0, 1>(tensor);
    const T& t10 = get<1, 0>(tensor);
    const T& t11 = get<1, 1>(tensor);

    get(*det) = t00 * t11 - t01 * t10;
    const T one_over_det = make_with_value<T>(*det, 1.0) / get(*det);
    get<0, 0>(*inv) = t11 * one_over_det;
    get<0, 1>(*inv) = -t01 * one_over_det;
    get<1, 0>(*inv) = -t10 * one_over_det;
    get<1, 1>(*inv) = t00 * one_over_det;
  }
};

template <typename Index0>
struct DetAndInverseImpl<Symmetry<1, 1>, Index0, Index0,
                         Requires<Index0::dim == 2>> {
  template <typename T>
  static void apply(
      const gsl::not_null<Scalar<T>*> det,
      const gsl::not_null<
          Tensor<T, Symmetry<1, 1>, inverse_indices<Index0, Index0>>*>
          inv,
      const Tensor<T, Symmetry<1, 1>, tmpl::list<Index0, Index0>>&
          tensor) noexcept {
    const T& t00 = get<0, 0>(tensor);
    const T& t01 = get<0, 1>(tensor);
    const T& t11 = get<1, 1>(tensor);

    get(*det) = t00 * t11 - t01 * t01;
    const T one_over_det = make_with_value<T>(*det, 1.0) / get(*det);
    get<0, 0>(*inv) = t11 * one_over_det;
    get<0, 1>(*inv) = -t01 * one_over_det;
    get<1, 1>(*inv) = t00 * one_over_det;
  }
};

// The inverse of a 3x3 tensor is computed from Cramer's rule. By reusing some
// terms, the determinant is computed efficiently at the same time.
template <typename Index0, typename Index1>
struct DetAndInverseImpl<Symmetry<2, 1>, Index0, Index1,
                         Requires<Index0::dim == 3>> {
  template <typename T>
  static void apply(
      const gsl::not_null<Scalar<T>*> det,
      const gsl::not_null<
          Tensor<T, Symmetry<2, 1>, inverse_indices<Index0, Index1>>*>
          inv,
      const Tensor<T, Symmetry<2, 1>, tmpl::list<Index0, Index1>>&
          tensor) noexcept {
    const T& t00 = get<0, 0>(tensor);
    const T& t01 = get<0, 1>(tensor);
    const T& t02 = get<0, 2>(tensor);
    const T& t10 = get<1, 0>(tensor);
    const T& t11 = get<1, 1>(tensor);
    const T& t12 = get<1, 2>(tensor);
    const T& t20 = get<2, 0>(tensor);
    const T& t21 = get<2, 1>(tensor);
    const T& t22 = get<2, 2>(tensor);
    const T a = t11 * t22 - t12 * t21;
    const T b = t12 * t20 - t10 * t22;
    const T c = t10 * t21 - t11 * t20;

    get(*det) = t00 * a + t01 * b + t02 * c;
    const T one_over_det = make_with_value<T>(*det, 1.0) / get(*det);
    get<0, 0>(*inv) = a * one_over_det;
    get<0, 1>(*inv) = (t21 * t02 - t22 * t01) * one_over_det;
    get<0, 2>(*inv) = (t01 * t12 - t02 * t11) * one_over_det;
    get<1, 0>(*inv) = b * one_over_det;
    get<1, 1>(*inv) = (t22 * t00 - t20 * t02) * one_over_det;
    get<1, 2>(*inv) = (t02 * t10 - t00 * t12) * one_over_det;
    get<2, 0>(*inv) = c * one_over_det;
    get<2, 1>(*inv) = (t20 * t01 - t21 * t00) * one_over_det;
    get<2, 2>(*inv) = (t00 * t11 - t01 * t10) * one_over_det;
  }
};

template <typename Index0>
struct DetAndInverseImpl<Symmetry<1, 1>, Index0, Index0,
                         Requires<Index0::dim == 3>> {
  template <typename T>
  static void apply(
      const gsl::not_null<Scalar<T>*> det,
      const gsl::not_null<
          Tensor<T, Symmetry<1, 1>, inverse_indices<Index0, Index0>>*>
          inv,
      const Tensor<T, Symmetry<1, 1>, tmpl::list<Index0, Index0>>&
          tensor) noexcept {
    const T& t00 = get<0, 0>(tensor);
    const T& t01 = get<0, 1>(tensor);
    const T& t02 = get<0, 2>(tensor);
    const T& t11 = get<1, 1>(tensor);
    const T& t12 = get<1, 2>(tensor);
    const T& t22 = get<2, 2>(tensor);
    const T a = t11 * t22 - t12 * t12;
    const T b = t12 * t02 - t01 * t22;
    const T c = t01 * t12 - t11 * t02;

    get(*det) = t00 * a + t01 * b + t02 * c;
    const T one_over_det = make_with_value<T>(*det, 1.0) / get(*det);
    get<0, 0>(*inv) = (t11 * t22 - t12 * t12) * one_over_det;
    get<0, 1>(*inv) = (t12 * t02 - t22 * t01) * one_over_det;
    get<0, 2>(*inv) = (t01 * t12 - t02 * t11) * one_over_det;
    get<1, 1>(*inv) = (t22 * t00 - t02 * t02) * one_over_det;
    get<1, 2>(*inv) = (t02 * t01 - t00 * t12) * one_over_det;
    get<2, 2>(*inv) = (t00 * t11 - t01 * t01) * one_over_det;
  }
};

// The 4x4 tensor inverse is implemented using the formula for inverting a
// partitioned matrix (here, partitioned into 2x2 blocks). This is more
// efficient than Cramer's rule for matrices larger than 3x3.
//
// In this algorithm, the 4x4 input matrix is partitioned into the 2x2 blocks:
//   P Q
//   R S
// The 4x4 inverse matrix is partitioned into the 2x2 blocks:
//   U V
//   W X
// Each inverse block is obtained from the blocks {P,Q,R,S} of the input matrix
// as follows:
//   X = inv[S - R inv(P) Q] (i.e. inv(X) is the Schur complement of P)
//   W = - X R inv(P)
//   V = - inv(P) Q X
//   U = inv(P) + inv(P) Q X R inv(P) = inv(P) - inv(P) Q W
template <typename Index0, typename Index1>
struct DetAndInverseImpl<Symmetry<2, 1>, Index0, Index1,
                         Requires<Index0::dim == 4>> {
  template <typename T>
  static void apply(
      const gsl::not_null<Scalar<T>*> det,
      const gsl::not_null<
          Tensor<T, Symmetry<2, 1>, inverse_indices<Index0, Index1>>*>
          inv,
      const Tensor<T, Symmetry<2, 1>, tmpl::list<Index0, Index1>>&
          tensor) noexcept {
    const T& p00 = get<0, 0>(tensor);
    const T& p01 = get<0, 1>(tensor);
    const T& p10 = get<1, 0>(tensor);
    const T& p11 = get<1, 1>(tensor);
    const T& q00 = get<0, 2>(tensor);
    const T& q01 = get<0, 3>(tensor);
    const T& q10 = get<1, 2>(tensor);
    const T& q11 = get<1, 3>(tensor);
    const T& r00 = get<2, 0>(tensor);
    const T& r01 = get<2, 1>(tensor);
    const T& r10 = get<3, 0>(tensor);
    const T& r11 = get<3, 1>(tensor);
    const T& s00 = get<2, 2>(tensor);
    const T& s01 = get<2, 3>(tensor);
    const T& s10 = get<3, 2>(tensor);
    const T& s11 = get<3, 3>(tensor);

    T& u00 = get<0, 0>(*inv);
    T& u01 = get<0, 1>(*inv);
    T& u10 = get<1, 0>(*inv);
    T& u11 = get<1, 1>(*inv);
    T& v00 = get<0, 2>(*inv);
    T& v01 = get<0, 3>(*inv);
    T& v10 = get<1, 2>(*inv);
    T& v11 = get<1, 3>(*inv);
    T& w00 = get<2, 0>(*inv);
    T& w01 = get<2, 1>(*inv);
    T& w10 = get<3, 0>(*inv);
    T& w11 = get<3, 1>(*inv);
    T& x00 = get<2, 2>(*inv);
    T& x01 = get<2, 3>(*inv);
    T& x10 = get<3, 2>(*inv);
    T& x11 = get<3, 3>(*inv);

    // Temporarily store det(P) in det
    get(*det) = p00 * p11 - p01 * p10;
    const T& det_p = get(*det);
    const T one_over_det_p = make_with_value<T>(det_p, 1.0) / det_p;
    const T inv_p00 = p11 * one_over_det_p;
    const T inv_p01 = -p01 * one_over_det_p;
    const T inv_p10 = -p10 * one_over_det_p;
    const T inv_p11 = p00 * one_over_det_p;

    const T r_inv_p00 = r00 * inv_p00 + r01 * inv_p10;
    const T r_inv_p01 = r00 * inv_p01 + r01 * inv_p11;
    const T r_inv_p10 = r10 * inv_p00 + r11 * inv_p10;
    const T r_inv_p11 = r10 * inv_p01 + r11 * inv_p11;

    const T inv_p_q00 = inv_p00 * q00 + inv_p01 * q10;
    const T inv_p_q01 = inv_p00 * q01 + inv_p01 * q11;
    const T inv_p_q10 = inv_p10 * q00 + inv_p11 * q10;
    const T inv_p_q11 = inv_p10 * q01 + inv_p11 * q11;

    const T inv_x00 = s00 - (r_inv_p00 * q00 + r_inv_p01 * q10);
    const T inv_x01 = s01 - (r_inv_p00 * q01 + r_inv_p01 * q11);
    const T inv_x10 = s10 - (r_inv_p10 * q00 + r_inv_p11 * q10);
    const T inv_x11 = s11 - (r_inv_p10 * q01 + r_inv_p11 * q11);

    const T det_inv_x = inv_x00 * inv_x11 - inv_x01 * inv_x10;
    const T one_over_det_inv_x = make_with_value<T>(det_inv_x, 1.0) / det_inv_x;
    x00 = inv_x11 * one_over_det_inv_x;
    x01 = -inv_x01 * one_over_det_inv_x;
    x10 = -inv_x10 * one_over_det_inv_x;
    x11 = inv_x00 * one_over_det_inv_x;

    w00 = -x00 * r_inv_p00 - x01 * r_inv_p10;
    w01 = -x00 * r_inv_p01 - x01 * r_inv_p11;
    w10 = -x10 * r_inv_p00 - x11 * r_inv_p10;
    w11 = -x10 * r_inv_p01 - x11 * r_inv_p11;

    v00 = -inv_p_q00 * x00 - inv_p_q01 * x10;
    v01 = -inv_p_q00 * x01 - inv_p_q01 * x11;
    v10 = -inv_p_q10 * x00 - inv_p_q11 * x10;
    v11 = -inv_p_q10 * x01 - inv_p_q11 * x11;

    u00 = inv_p00 - (inv_p_q00 * w00 + inv_p_q01 * w10);
    u01 = inv_p01 - (inv_p_q00 * w01 + inv_p_q01 * w11);
    u10 = inv_p10 - (inv_p_q10 * w00 + inv_p_q11 * w10);
    u11 = inv_p11 - (inv_p_q10 * w01 + inv_p_q11 * w11);

    get(*det) *= det_inv_x;
  }
};

template <typename Index0>
struct DetAndInverseImpl<Symmetry<1, 1>, Index0, Index0,
                         Requires<Index0::dim == 4>> {
  template <typename T>
  static void apply(
      const gsl::not_null<Scalar<T>*> det,
      const gsl::not_null<
          Tensor<T, Symmetry<1, 1>, inverse_indices<Index0, Index0>>*>
          inv,
      const Tensor<T, Symmetry<1, 1>, tmpl::list<Index0, Index0>>&
          tensor) noexcept {
    const T& p00 = get<0, 0>(tensor);
    const T& p01 = get<0, 1>(tensor);
    const T& p11 = get<1, 1>(tensor);
    const T& q00 = get<0, 2>(tensor);
    const T& q01 = get<0, 3>(tensor);
    const T& q10 = get<1, 2>(tensor);
    const T& q11 = get<1, 3>(tensor);
    const T& s00 = get<2, 2>(tensor);
    const T& s01 = get<2, 3>(tensor);
    const T& s11 = get<3, 3>(tensor);

    T& u00 = get<0, 0>(*inv);
    T& u01 = get<0, 1>(*inv);
    T& u11 = get<1, 1>(*inv);
    T& v00 = get<0, 2>(*inv);
    T& v01 = get<0, 3>(*inv);
    T& v10 = get<1, 2>(*inv);
    T& v11 = get<1, 3>(*inv);
    T& x00 = get<2, 2>(*inv);
    T& x01 = get<2, 3>(*inv);
    T& x11 = get<3, 3>(*inv);

    // Temporarily store det(P) in det
    get(*det) = p00 * p11 - p01 * p01;
    const T& det_p = get(*det);
    const T one_over_det_p = make_with_value<T>(det_p, 1.0) / det_p;
    const T inv_p00 = p11 * one_over_det_p;
    const T inv_p01 = -p01 * one_over_det_p;
    const T inv_p11 = p00 * one_over_det_p;

    const T r_inv_p00 = q00 * inv_p00 + q10 * inv_p01;
    const T r_inv_p01 = q00 * inv_p01 + q10 * inv_p11;
    const T r_inv_p10 = q01 * inv_p00 + q11 * inv_p01;
    const T r_inv_p11 = q01 * inv_p01 + q11 * inv_p11;

    const T inv_x00 = s00 - (r_inv_p00 * q00 + r_inv_p01 * q10);
    const T inv_x01 = s01 - (r_inv_p00 * q01 + r_inv_p01 * q11);
    const T inv_x11 = s11 - (r_inv_p10 * q01 + r_inv_p11 * q11);

    const T det_inv_x = inv_x00 * inv_x11 - inv_x01 * inv_x01;
    const T one_over_det_inv_x = make_with_value<T>(det_inv_x, 1.0) / det_inv_x;
    x00 = inv_x11 * one_over_det_inv_x;
    x01 = -inv_x01 * one_over_det_inv_x;
    x11 = inv_x00 * one_over_det_inv_x;

    v00 = -x00 * r_inv_p00 - x01 * r_inv_p10;
    v10 = -x00 * r_inv_p01 - x01 * r_inv_p11;
    v01 = -x01 * r_inv_p00 - x11 * r_inv_p10;
    v11 = -x01 * r_inv_p01 - x11 * r_inv_p11;

    u00 = inv_p00 - (r_inv_p00 * v00 + r_inv_p10 * v01);
    u01 = inv_p01 - (r_inv_p00 * v10 + r_inv_p10 * v11);
    u11 = inv_p11 - (r_inv_p01 * v10 + r_inv_p11 * v11);

    get(*det) *= det_inv_x;
  }
};
}  // namespace determinant_and_inverse_detail

// @{
/*!
 * \ingroup TensorGroup
 * \brief Computes the determinant and inverse of a rank-2 Tensor.
 *
 * Computes the determinant and inverse together, because this leads to
 * fewer operations compared to computing the determinant independently.
 *
 * \details
 * Treats the input rank-2 tensor as a matrix. The first (second) index
 * of the tensor corresponds to the rows (columns) of the matrix. The
 * determinant is a scalar tensor. The inverse is a rank-2 tensor whose
 * indices are reversed and of opposite valence relative to the input
 * tensor, i.e. given \f$T_a^b\f$ returns \f$(Tinv)_b^a\f$.
 *
 * \note
 * When inverting a 4x4 spacetime metric, it is typically more efficient
 * to use the 3+1 decomposition of the 4-metric in terms of lapse,
 * shift, and spatial 3-metric, in which only the spatial 3-metric needs
 * to be inverted.
 */
template <typename T, typename Symm, typename Index0, typename Index1>
void determinant_and_inverse(
    const gsl::not_null<Scalar<T>*> det,
    const gsl::not_null<Tensor<
        T, Symm,
        tmpl::list<change_index_up_lo<Index1>, change_index_up_lo<Index0>>>*>
        inv,
    const Tensor<T, Symm, tmpl::list<Index0, Index1>>& tensor) noexcept {
  static_assert(Index0::dim == Index1::dim,
                "Cannot take the inverse of a Tensor whose Indices are not "
                "of the same dimensionality.");
  static_assert(Index0::index_type == Index1::index_type,
                "Taking the inverse of a mixed Spatial and Spacetime index "
                "Tensor is not allowed since it's not clear what that means.");
  static_assert(not std::is_integral<T>::value, "Can't invert a Tensor<int>.");

  if constexpr (not tt::is_complex_or_fundamental_v<T>) {
    destructive_resize_components(det, get<0, 0>(tensor).size());
    destructive_resize_components(inv, get<0, 0>(tensor).size());
  }
  determinant_and_inverse_detail::DetAndInverseImpl<Symm, Index0,
                                                    Index1>::apply(det, inv,
                                                                   tensor);
}

template <typename T, typename Symm, typename Index0, typename Index1>
auto determinant_and_inverse(
    const Tensor<T, Symm, tmpl::list<Index0, Index1>>& tensor) noexcept
    -> std::pair<Scalar<T>, Tensor<T, Symm,
                                   tmpl::list<change_index_up_lo<Index1>,
                                              change_index_up_lo<Index0>>>> {
  std::pair<Scalar<T>, Tensor<T, Symm,
                              tmpl::list<change_index_up_lo<Index1>,
                                         change_index_up_lo<Index0>>>>
      result{};
  determinant_and_inverse_detail::DetAndInverseImpl<
      Symm, Index0, Index1>::apply(make_not_null(&result.first),
                                   make_not_null(&result.second), tensor);
  return result;
}
// @}

/*!
 * \ingroup TensorGroup
 * \brief Computes the determinant and inverse of a rank-2 Tensor.
 *
 * Computes the determinant and inverse together, because this leads to fewer
 * operations compared to computing the determinant independently.
 *
 * \param tensor the input rank-2 Tensor.
 * \tparam DetTag the Tag for the determinant of input Tensor.
 * \tparam InvTag the Tag for the inverse of input Tensor.
 * \return a `Variables` that holds the determinant and inverse of the input
 * tensor, tagged by `DetTag` and `InvTag` respectively.
 *
 * \details
 * See determinant_and_inverse().
 */
template <typename DetTag, typename InvTag, typename T, typename Symm,
          typename Index0, typename Index1>
auto determinant_and_inverse(
    const Tensor<T, Symm, tmpl::list<Index0, Index1>>& tensor) noexcept
    -> Variables<tmpl::list<DetTag, InvTag>> {
  static_assert(std::is_same_v<typename DetTag::type, Scalar<T>>,
                "Type of first return tag must correspond to that of input's "
                "determinant.");
  static_assert(
      std::is_same_v<typename InvTag::type,
                     Tensor<T, Symm,
                            tmpl::list<change_index_up_lo<Index1>,
                                       change_index_up_lo<Index0>>>>,
      "Type of second return tag must correspond to that of input's inverse.");
  Variables<tmpl::list<DetTag, InvTag>> result(get<0, 0>(tensor).size());
  determinant_and_inverse_detail::DetAndInverseImpl<
      Symm, Index0, Index1>::apply(make_not_null(&get<DetTag>(result)),
                                   make_not_null(&get<InvTag>(result)), tensor);
  return result;
}
