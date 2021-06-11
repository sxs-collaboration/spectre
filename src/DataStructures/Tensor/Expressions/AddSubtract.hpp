// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines ET for adding and subtracting tensors

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/NumberAsExpression.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

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
          std::conditional_t<
              std::is_same<typename T1::type, DataVector>::value or
                  std::is_same<typename T2::type, DataVector>::value,
              DataVector, double>,
          tmpl::transform<typename T1::symmetry, typename T2::symmetry,
                          tmpl::append<tmpl::max<tmpl::_1, tmpl::_2>>>,
          typename T1::index_list, typename T1::args_list> {
  static_assert(std::is_same<typename T1::type, typename T2::type>::value or
                    std::is_same<T1, NumberAsExpression>::value or
                    std::is_same<T2, NumberAsExpression>::value,
                "Cannot add or subtract Tensors holding different data types.");
  static_assert(
      detail::AddSubIndexCheck<typename T1::index_list, typename T2::index_list,
                               ArgsList1<Args1...>, ArgsList2<Args2...>>::value,
      "You are attempting to add indices of different types, e.g. T^a_b + "
      "S^b_a, which doesn't make sense. The indices may also be in different "
      "frames, different types (spatial vs. spacetime) or of different "
      "dimension.");
  static_assert(Sign == 1 or Sign == -1,
                "Invalid Sign provided for addition or subtraction of Tensor "
                "elements. Sign must be 1 (addition) or -1 (subtraction).");

  using type =
      std::conditional_t<std::is_same<typename T1::type, DataVector>::value or
                             std::is_same<typename T2::type, DataVector>::value,
                         DataVector, double>;
  using symmetry = tmpl::transform<typename T1::symmetry, typename T2::symmetry,
                                   tmpl::append<tmpl::max<tmpl::_1, tmpl::_2>>>;
  using index_list = typename T1::index_list;
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;
  using args_list = typename T1::args_list;

  AddSub(T1 t1, T2 t2) : t1_(std::move(t1)), t2_(std::move(t2)) {}
  ~AddSub() override = default;

  template <typename... LhsIndices>
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<size_t, num_tensor_indices>& lhs_multi_index) const {
    if constexpr (Sign == 1) {
      return t1_.template get<LhsIndices...>(lhs_multi_index) +
             t2_.template get<LhsIndices...>(lhs_multi_index);
    } else {
      return t1_.template get<LhsIndices...>(lhs_multi_index) -
             t2_.template get<LhsIndices...>(lhs_multi_index);
    }
  }

  SPECTRE_ALWAYS_INLINE typename T1::type operator[](size_t i) const {
    if constexpr (Sign == 1) {
      return t1_[i] + t2_[i];
    } else {
      return t1_[i] - t2_[i];
    }
  }

 private:
  T1 t1_;
  T2 t2_;
};
}  // namespace TensorExpressions

/*!
 * \ingroup TensorExpressionsGroup
 */
template <typename T1, typename T2, typename X1, typename X2, typename Symm1,
          typename Symm2, typename IndexList1, typename IndexList2,
          typename Args1, typename Args2>
SPECTRE_ALWAYS_INLINE auto operator+(
    const TensorExpression<T1, X1, Symm1, IndexList1, Args1>& t1,
    const TensorExpression<T2, X2, Symm2, IndexList2, Args2>& t2) {
  static_assert(tmpl::size<Args1>::value == tmpl::size<Args2>::value,
                "Tensor addition is only possible with the same rank tensors");
  static_assert(tmpl::equal_members<Args1, Args2>::value,
                "The indices when adding two tensors must be equal. This error "
                "occurs from expressions like A(_a, _b) + B(_c, _a)");
  return TensorExpressions::AddSub<T1, T2, Args1, Args2, 1>(~t1, ~t2);
}

/// @{
/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the sum of a tensor
/// expression and a `double`
///
/// \details
/// The tensor expression operand must represent an expression that, when
/// evaluated, would be a rank 0 tensor. For example, if `R` and `S` are
/// Tensors, here is a non-exhaustive list of some of the acceptable forms that
/// the tensor expression operand could take:
/// - `R()`
/// - `R(ti_A, ti_a)`
/// - `(R(ti_A, ti_B) * S(ti_a, ti_b))`
///
/// \tparam T the derived TensorExpression type of the tensor expression operand
/// of the sum
/// \tparam X the type of data stored in the tensor expression operand of the
/// sum
/// \param t the tensor expression operand of the sum
/// \param number the `double` operand of the sum
/// \return the tensor expression representing the sum of a tensor expression
/// and a `double`
template <typename T, typename X>
SPECTRE_ALWAYS_INLINE auto operator+(
    const TensorExpression<T, X, tmpl::list<>, tmpl::list<>, tmpl::list<>>& t,
    const double number) {
  return t + TensorExpressions::NumberAsExpression(number);
}
template <typename T, typename X>
SPECTRE_ALWAYS_INLINE auto operator+(
    const double number,
    const TensorExpression<T, X, tmpl::list<>, tmpl::list<>, tmpl::list<>>& t) {
  return TensorExpressions::NumberAsExpression(number) + t;
}
/// @}

/*!
 * \ingroup TensorExpressionsGroup
 */
template <typename T1, typename T2, typename X1, typename X2, typename Symm1,
          typename Symm2, typename IndexList1, typename IndexList2,
          typename Args1, typename Args2>
SPECTRE_ALWAYS_INLINE auto operator-(
    const TensorExpression<T1, X1, Symm1, IndexList1, Args1>& t1,
    const TensorExpression<T2, X2, Symm2, IndexList2, Args2>& t2) {
  static_assert(tmpl::size<Args1>::value == tmpl::size<Args2>::value,
                "Tensor addition is only possible with the same rank tensors");
  static_assert(tmpl::equal_members<Args1, Args2>::value,
                "The indices when adding two tensors must be equal. This error "
                "occurs from expressions like A(_a, _b) - B(_c, _a)");
  return TensorExpressions::AddSub<T1, T2, Args1, Args2, -1>(~t1, ~t2);
}

/// @{
/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the difference of a tensor
/// expression and a `double`
///
/// \details
/// The tensor expression operand must represent an expression that, when
/// evaluated, would be a rank 0 tensor. For example, if `R` and `S` are
/// Tensors, here is a non-exhaustive list of some of the acceptable forms that
/// the tensor expression operand could take:
/// - `R()`
/// - `R(ti_A, ti_a)`
/// - `(R(ti_A, ti_B) * S(ti_a, ti_b))`
///
/// \tparam T the derived TensorExpression type of the tensor expression operand
/// of the difference
/// \tparam X the type of data stored in the tensor expression operand of the
/// difference
/// \param t the tensor expression operand of the difference
/// \param number the `double` operand of the difference
/// \return the tensor expression representing the difference of a tensor
/// expression and a `double`
template <typename T, typename X>
SPECTRE_ALWAYS_INLINE auto operator-(
    const TensorExpression<T, X, tmpl::list<>, tmpl::list<>, tmpl::list<>>& t,
    const double number) {
  return t - TensorExpressions::NumberAsExpression(number);
}
template <typename T, typename X>
SPECTRE_ALWAYS_INLINE auto operator-(
    const double number,
    const TensorExpression<T, X, tmpl::list<>, tmpl::list<>, tmpl::list<>>& t) {
  return TensorExpressions::NumberAsExpression(number) - t;
}
/// @}
