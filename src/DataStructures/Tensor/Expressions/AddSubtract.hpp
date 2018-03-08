// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines ET for adding and subtracting tensors

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"
/// \cond
namespace TensorExpressions {
template <typename T1, typename T2, typename ArgsList1, typename ArgsList2,
          int Sign>
struct AddSub;
}  // namespace TensorExpressions
template <typename Derived, typename DataType, typename Symm,
          typename IndexList, typename Args, typename ReducedArgs>
struct TensorExpression;
/// \endcond

namespace TensorExpressions {

namespace detail {
template <typename IndexList1, typename IndexList2, typename Args1,
          typename Args2, typename Element>
struct AddSubIndexCheckHelper
    : std::is_same<tmpl::at<IndexList1, tmpl::index_of<Args1, Element>>,
                   tmpl::at<IndexList2, tmpl::index_of<Args2, Element>>>::type {
};

// Check to make sure that the tensor indices being added are of the same type,
// dimensionality and in the same frame
template <typename IndexList1, typename IndexList2, typename Args1,
          typename Args2>
using AddSubIndexCheck = tmpl::fold<
    Args1, tmpl::bool_<true>,
    tmpl::and_<tmpl::_state,
               AddSubIndexCheckHelper<tmpl::pin<IndexList1>,
                                      tmpl::pin<IndexList2>, tmpl::pin<Args1>,
                                      tmpl::pin<Args2>, tmpl::_element>>>;
}  // namespace detail

template <typename T1, typename T2, typename ArgsList1, typename ArgsList2,
          int Sign>
struct AddSub;

template <typename T1, typename T2, template <typename...> class ArgsList1,
          template <typename...> class ArgsList2, typename... Args1,
          typename... Args2, int Sign>
struct AddSub<T1, T2, ArgsList1<Args1...>, ArgsList2<Args2...>, Sign>
    : public TensorExpression<
          AddSub<T1, T2, ArgsList1<Args1...>, ArgsList2<Args2...>, Sign>,
          typename T1::type,
          tmpl::transform<typename T1::symmetry, typename T2::symmetry,
                          tmpl::append<tmpl::max<tmpl::_1, tmpl::_2>>>,
          typename T1::index_list, tmpl::sort<typename T1::args_list>>,
      public Expression {
  static_assert(std::is_same<typename T1::type, typename T2::type>::value,
                "Cannot add or subtract Tensors holding different data types.");
  static_assert(
      detail::AddSubIndexCheck<typename T1::index_list, typename T2::index_list,
                               ArgsList1<Args1...>, ArgsList2<Args2...>>::value,
      "You are attempting to add indices of different types, e.g. T^a_b + "
      "S^b_a, which doesn't make sense. The indices may also be in different "
      "frames, different types (spatial vs. spacetime) or of different "
      "dimension.");

  using type = typename T1::type;
  using symmetry = tmpl::transform<typename T1::symmetry, typename T2::symmetry,
                                   tmpl::append<tmpl::max<tmpl::_1, tmpl::_2>>>;
  using index_list = typename T1::index_list;
  static constexpr auto num_tensor_indices =
      tmpl::size<index_list>::value == 0 ? 1 : tmpl::size<index_list>::value;
  using args_list = tmpl::sort<typename T1::args_list>;

  AddSub(T1 t1, T2 t2) : t1_(std::move(t1)), t2_(std::move(t2)) {}

  template <typename... LhsIndices, typename T, int U = Sign,
            Requires<U == 1> = nullptr>
  SPECTRE_ALWAYS_INLINE type
  get(const std::array<T, num_tensor_indices>& tensor_index) const {
    return t1_.template get<LhsIndices...>(tensor_index) +
           t2_.template get<LhsIndices...>(tensor_index);
  }

  template <typename... LhsIndices, typename T, int U = Sign,
            Requires<U == -1> = nullptr>
  SPECTRE_ALWAYS_INLINE type
  get(const std::array<T, num_tensor_indices>& tensor_index) const {
    return t1_.template get<LhsIndices...>(tensor_index) -
           t2_.template get<LhsIndices...>(tensor_index);
  }

  template <int U = Sign, Requires<U == 1> = nullptr>
  SPECTRE_ALWAYS_INLINE typename T1::type operator[](size_t i) const {
    return t1_[i] + t2_[i];
  }

  template <int U = Sign, Requires<U == -1> = nullptr>
  SPECTRE_ALWAYS_INLINE typename T1::type operator[](size_t i) const {
    return t1_[i] - t2_[i];
  }

 private:
  const T1 t1_;
  const T2 t2_;
};
}  // namespace TensorExpressions

/*!
 * \ingroup TensorExpressionsGroup
 */
template <typename T1, typename T2, typename X, typename Symm1, typename Symm2,
          typename IndexList1, typename IndexList2, typename Args1,
          typename Args2>
SPECTRE_ALWAYS_INLINE auto operator+(
    const TensorExpression<T1, X, Symm1, IndexList1, Args1>& t1,
    const TensorExpression<T2, X, Symm2, IndexList2, Args2>& t2) {
  static_assert(tmpl::size<Args1>::value == tmpl::size<Args2>::value,
                "Tensor addition is only possible with the same rank tensors");
  static_assert(tmpl::equal_members<Args1, Args2>::value,
                "The indices when adding two tensors must be equal. This error "
                "occurs from expressions like A(_a, _b) + B(_c, _a)");
  return TensorExpressions::AddSub<
      tmpl::conditional_t<std::is_base_of<Expression, T1>::value, T1,
                          TensorExpression<T1, X, Symm1, IndexList1, Args1>>,
      tmpl::conditional_t<std::is_base_of<Expression, T2>::value, T2,
                          TensorExpression<T2, X, Symm2, IndexList2, Args2>>,
      Args1, Args2, 1>(~t1, ~t2);
}

/*!
 * \ingroup TensorExpressionsGroup
 */
template <typename T1, typename T2, typename X, typename Symm1, typename Symm2,
          typename IndexList1, typename IndexList2, typename Args1,
          typename Args2>
SPECTRE_ALWAYS_INLINE auto operator-(
    const TensorExpression<T1, X, Symm1, IndexList1, Args1>& t1,
    const TensorExpression<T2, X, Symm2, IndexList2, Args2>& t2) {
  static_assert(tmpl::size<Args1>::value == tmpl::size<Args2>::value,
                "Tensor addition is only possible with the same rank tensors");
  static_assert(tmpl::equal_members<Args1, Args2>::value,
                "The indices when adding two tensors must be equal. This error "
                "occurs from expressions like A(_a, _b) - B(_c, _a)");
  return TensorExpressions::AddSub<
      tmpl::conditional_t<std::is_base_of<Expression, T1>::value, T1,
                          TensorExpression<T1, X, Symm1, IndexList1, Args1>>,
      tmpl::conditional_t<std::is_base_of<Expression, T2>::value, T2,
                          TensorExpression<T2, X, Symm2, IndexList2, Args2>>,
      Args1, Args2, -1>(~t1, ~t2);
}
