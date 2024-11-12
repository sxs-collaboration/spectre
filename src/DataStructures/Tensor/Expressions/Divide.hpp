// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines ET for tensor division by scalars

#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

#include "DataStructures/Tensor/Expressions/DataTypeSupport.hpp"
#include "DataStructures/Tensor/Expressions/NumberAsExpression.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Expressions/TimeIndex.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace tenex {
/// \ingroup TensorExpressionsGroup
/// \brief Defines the tensor expression representing the quotient of one tensor
/// expression divided by another tensor expression that evaluates to a rank 0
/// tensor
///
/// \details
/// For details on aliases and members defined in this class, as well as general
/// `TensorExpression` terminology used in its members' documentation, see
/// documentation for `TensorExpression`.
///
/// \tparam T1 the numerator operand expression of the division expression
/// \tparam T2 the denominator operand expression of the division expression
/// \tparam Args2 the generic indices of the denominator expression
template <typename T1, typename T2, typename... Args2>
struct Divide
    : public TensorExpression<Divide<T1, T2, Args2...>,
                              typename detail::get_binop_datatype<
                                  typename T1::type, typename T2::type>::type,
                              typename T1::symmetry, typename T1::index_list,
                              typename T1::args_list> {
  static_assert(
      detail::tensorexpression_binop_datatypes_are_supported_v<T1, T2>,
      "Cannot divide the given TensorExpressions with the given data types. "
      "This can occur from e.g. trying to divide a Tensor with data type "
      "double and a Tensor with data type DataVector.");
  static_assert(
      not((std::is_same_v<T1, NumberAsExpression<std::complex<double>>> and
           std::is_same_v<typename T2::type, DataVector>) or
          (std::is_same_v<T2, NumberAsExpression<std::complex<double>>> and
           std::is_same_v<typename T1::type, DataVector>)),
      "Cannot perform division between a std::complex<double> and a "
      "TensorExpression whose data type is DataVector because Blaze does not "
      "support division between std::complex<double> and DataVector.");
  static_assert((... and tt::is_time_index<Args2>::value),
                "Can only divide a tensor expression by a number or a tensor "
                "expression that evaluates to "
                "a rank 0 tensor.");

  // === Index properties ===
  /// The type of the data being stored in the result of the expression
  using type = typename detail::get_binop_datatype<typename T1::type,
                                                   typename T2::type>::type;
  /// The ::Symmetry of the result of the expression
  using symmetry = typename T1::symmetry;
  /// The list of \ref SpacetimeIndex "TensorIndexType"s of the result of the
  /// expression
  using index_list = typename T1::index_list;
  /// The list of generic `TensorIndex`s of the result of the expression
  using args_list = typename T1::args_list;
  /// The number of tensor indices in the result of the expression
  static constexpr auto num_tensor_indices =
      tmpl::size<typename T1::index_list>::value;
  /// The number of tensor indices in the left operand expression
  static constexpr auto op2_num_tensor_indices =
      tmpl::size<typename T2::index_list>::value;
  /// The multi-index for the denominator
  static constexpr auto op2_multi_index =
      make_array<op2_num_tensor_indices, size_t>(0);

  // === Expression subtree properties ===
  /// The number of arithmetic tensor operations done in the subtree for the
  /// left operand
  static constexpr size_t num_ops_left_child = T1::num_ops_subtree;
  /// The number of arithmetic tensor operations done in the subtree for the
  /// right operand
  static constexpr size_t num_ops_right_child = T2::num_ops_subtree;
  /// The total number of arithmetic tensor operations done in this expression's
  /// whole subtree
  static constexpr size_t num_ops_subtree =
      num_ops_left_child + num_ops_right_child + 1;
  /// The height of this expression's node in the expression tree relative to
  /// the closest `TensorAsExpression` leaf in its subtree
  static constexpr size_t height_relative_to_closest_tensor_leaf_in_subtree =
      T2::height_relative_to_closest_tensor_leaf_in_subtree <=
              T1::height_relative_to_closest_tensor_leaf_in_subtree
          ? (T2::height_relative_to_closest_tensor_leaf_in_subtree !=
                     std::numeric_limits<size_t>::max()
                 ? T2::height_relative_to_closest_tensor_leaf_in_subtree + 1
                 : T2::height_relative_to_closest_tensor_leaf_in_subtree)
          : T1::height_relative_to_closest_tensor_leaf_in_subtree + 1;

  // === Properties for splitting up subexpressions along the primary path ===
  // These definitions only have meaning if this expression actually ends up
  // being along the primary path that is taken when evaluating the whole tree.
  // See documentation for `TensorExpression` for more details.
  /// If on the primary path, whether or not the expression is an ending point
  /// of a leg
  static constexpr bool is_primary_end = T1::is_primary_start;
  /// If on the primary path, this is the remaining number of arithmetic tensor
  /// operations that need to be done in the subtree of the child along the
  /// primary path, given that we will have already computed the whole subtree
  /// at the next lowest leg's starting point.
  static constexpr size_t num_ops_to_evaluate_primary_left_child =
      is_primary_end ? 0 : T1::num_ops_to_evaluate_primary_subtree;
  /// If on the primary path, this is the remaining number of arithmetic tensor
  /// operations that need to be done in the right operand's subtree. No
  /// splitting is currently done, so this is just `num_ops_right_child`.
  static constexpr size_t num_ops_to_evaluate_primary_right_child =
      num_ops_right_child;
  /// If on the primary path, this is the remaining number of arithmetic tensor
  /// operations that need to be done for this expression's subtree, given that
  /// we will have already computed the subtree at the next lowest leg's
  /// starting point
  static constexpr size_t num_ops_to_evaluate_primary_subtree =
      num_ops_to_evaluate_primary_left_child +
      num_ops_to_evaluate_primary_right_child + 1;
  /// If on the primary path, whether or not the expression is a starting point
  /// of a leg
  static constexpr bool is_primary_start =
      num_ops_to_evaluate_primary_subtree >=
      detail::max_num_ops_in_sub_expression<type>;
  /// When evaluating along a primary path, whether each operand's subtrees
  /// should be evaluated separately. Since `DataVector` expression runtime
  /// scales poorly with increased number of operations, evaluating the two
  /// expression subtrees separately like this is beneficial when at least one
  /// of the subtrees contains a large number of operations.
  static constexpr bool evaluate_children_separately =
      is_primary_start and (num_ops_to_evaluate_primary_left_child >=
                                detail::max_num_ops_in_sub_expression<type> or
                            num_ops_to_evaluate_primary_right_child >=
                                detail::max_num_ops_in_sub_expression<type>);
  /// If on the primary path, whether or not the expression's child along the
  /// primary path is a subtree that contains a starting point of a leg along
  /// the primary path
  static constexpr bool primary_child_subtree_contains_primary_start =
      T1::primary_subtree_contains_primary_start;
  /// If on the primary path, whether or not this subtree contains a starting
  /// point of a leg along the primary path
  static constexpr bool primary_subtree_contains_primary_start =
      is_primary_start or primary_child_subtree_contains_primary_start;

  Divide(T1 t1, T2 t2) : t1_(std::move(t1)), t2_(std::move(t2)) {}
  ~Divide() override = default;

  /// \brief Assert that the LHS tensor of the equation does not also appear in
  /// this expression's subtree
  template <typename LhsTensor>
  SPECTRE_ALWAYS_INLINE void assert_lhs_tensor_not_in_rhs_expression(
      const gsl::not_null<LhsTensor*> lhs_tensor) const {
    if constexpr (not std::is_base_of_v<MarkAsNumberAsExpression, T1>) {
      t1_.assert_lhs_tensor_not_in_rhs_expression(lhs_tensor);
    }
    if constexpr (not std::is_base_of_v<MarkAsNumberAsExpression, T2>) {
      t2_.assert_lhs_tensor_not_in_rhs_expression(lhs_tensor);
    }
  }

  /// \brief Assert that each instance of the LHS tensor in the RHS tensor
  /// expression uses the same generic index order that the LHS uses
  ///
  /// \tparam LhsTensorIndices the list of generic `TensorIndex`s of the LHS
  /// result `Tensor` being computed
  /// \param lhs_tensor the LHS result `Tensor` being computed
  template <typename LhsTensorIndices, typename LhsTensor>
  SPECTRE_ALWAYS_INLINE void assert_lhs_tensorindices_same_in_rhs(
      const gsl::not_null<LhsTensor*> lhs_tensor) const {
    if constexpr (not std::is_base_of_v<MarkAsNumberAsExpression, T1>) {
      t1_.assert_lhs_tensorindices_same_in_rhs(lhs_tensor);
    }
    if constexpr (not std::is_base_of_v<MarkAsNumberAsExpression, T2>) {
      t2_.assert_lhs_tensorindices_same_in_rhs(lhs_tensor);
    }
  }

  /// \brief Get the size of a component from a `Tensor` in this expression's
  /// subtree of the RHS `TensorExpression`
  ///
  /// \return the size of a component from a `Tensor` in this expression's
  /// subtree of the RHS `TensorExpression`
  SPECTRE_ALWAYS_INLINE size_t get_rhs_tensor_component_size() const {
    if constexpr (T1::height_relative_to_closest_tensor_leaf_in_subtree <=
                  T2::height_relative_to_closest_tensor_leaf_in_subtree) {
      return t1_.get_rhs_tensor_component_size();
    } else {
      return t2_.get_rhs_tensor_component_size();
    }
  }

  /// \brief Return the value of the component of the quotient tensor at a given
  /// multi-index
  ///
  /// \param result_multi_index the multi-index of the component of the quotient
  //// tensor to retrieve
  /// \return the value of the component in the quotient tensor at
  /// `result_multi_index`
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<size_t, num_tensor_indices>& result_multi_index) const {
    return t1_.get(result_multi_index) / t2_.get(op2_multi_index);
  }

  /// \brief Return the value of the component of the quotient tensor at a given
  /// multi-index
  ///
  /// \details
  /// This function differs from `get` in that it takes into account whether we
  /// have already computed part of the result component at a lower subtree.
  /// In recursively computing this quotient, the current result component will
  /// be substituted in for the most recent (highest) subtree below it that has
  /// already been evaluated.
  ///
  /// \param result_component the LHS tensor component to evaluate
  /// \param result_multi_index the multi-index of the component of the quotient
  //// tensor to retrieve
  /// \return the value of the component in the quotient tensor at
  /// `result_multi_index`
  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE decltype(auto) get_primary(
      const ResultType& result_component,
      const std::array<size_t, num_tensor_indices>& result_multi_index) const {
    if constexpr (is_primary_end) {
      (void)result_multi_index;
      // We've already computed the whole child subtree on the primary path, so
      // just return the quotient of the current result component and the result
      // of the other child's subtree
      return result_component / t2_.get(op2_multi_index);
    } else {
      // We haven't yet evaluated the whole subtree for this expression, so
      // return the quotient of the results of the two operands' subtrees
      return t1_.get_primary(result_component, result_multi_index) /
             t2_.get(op2_multi_index);
    }
  }

  /// \brief Evaluate the LHS Tensor's result component at this subtree by
  /// evaluating the two operand's subtrees separately and dividing
  ///
  /// \details
  /// This function takes into account whether we have already computed part of
  /// the result component at a lower subtree. In recursively computing this
  /// quotient, the current result component will be substituted in for the most
  /// recent (highest) subtree below it that has already been evaluated.
  ///
  /// The left and right operands' subtrees are evaluated successively with
  /// two separate assignments to the LHS result component. Since `DataVector`
  /// expression runtime scales poorly with increased number of operations,
  /// evaluating the two expression subtrees separately like this is beneficial
  /// when at least one of the subtrees contains a large number of operations.
  /// Instead of evaluating a larger expression with their combined total number
  /// of operations, we evaluate two smaller ones.
  ///
  /// \param result_component the LHS tensor component to evaluate
  /// \param result_multi_index the multi-index of the component of the result
  /// tensor to evaluate
  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE void evaluate_primary_children(
      ResultType& result_component,
      const std::array<size_t, num_tensor_indices>& result_multi_index) const {
    if constexpr (is_primary_end) {
      (void)result_multi_index;
      // We've already computed the whole child subtree on the primary path, so
      // just divide the current result by the result of the other child's
      // subtree
      result_component /= t2_.get(op2_multi_index);
    } else {
      // We haven't yet evaluated the whole subtree of the primary child, so
      // first assign the result component to be the result of computing the
      // primary child's subtree
      result_component = t1_.get_primary(result_component, result_multi_index);
      // Now that the primary child's subtree has been computed, divide the
      // current result by the result of evaluating the other child's subtree
      result_component /= t2_.get(op2_multi_index);
    }
  }

  /// \brief Successively evaluate the LHS Tensor's result component at each
  /// leg in this expression's subtree
  ///
  /// \details
  /// This function takes into account whether we have already computed part of
  /// the result component at a lower subtree. In recursively computing this
  /// quotient, the current result component will be substituted in for the most
  /// recent (highest) subtree below it that has already been evaluated.
  ///
  /// \param result_component the LHS tensor component to evaluate
  /// \param result_multi_index the multi-index of the component of the result
  /// tensor to evaluate
  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE void evaluate_primary_subtree(
      ResultType& result_component,
      const std::array<size_t, num_tensor_indices>& result_multi_index) const {
    if constexpr (primary_child_subtree_contains_primary_start) {
      // The primary child's subtree contains at least one leg, so recurse down
      // and evaluate that first
      t1_.evaluate_primary_subtree(result_component, result_multi_index);
    }
    if constexpr (is_primary_start) {
      // We want to evaluate the subtree for this expression
      if constexpr (evaluate_children_separately) {
        // Evaluate operand's subtrees separately
        evaluate_primary_children(result_component, result_multi_index);
      } else {
        // Evaluate whole subtree as one expression
        result_component = get_primary(result_component, result_multi_index);
      }
    }
  }

 private:
  /// Left operand (numerator)
  T1 t1_;
  /// Right operand (denominator)
  T2 t2_;
};
}  // namespace tenex

/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the quotient of one tensor
/// expression over another tensor expression that evaluates to a rank 0 tensor
///
/// \details
/// `t2` must be an expression that, when evaluated, would be a rank 0 tensor.
/// For example, if `R` and `S` are Tensors, here is a non-exhaustive list of
/// some of the acceptable forms that `t2` could take:
/// - `R()`
/// - `R(ti::A, ti::a)`
/// - `(R(ti::A, ti::B) * S(ti::a, ti::b))`
/// - `R(ti::t, ti::t) + 1.0`
///
/// \param t1 the tensor expression numerator
/// \param t2 the rank 0 tensor expression denominator
template <typename T1, typename T2, typename... Args2>
SPECTRE_ALWAYS_INLINE auto operator/(
    const TensorExpression<T1, typename T1::type, typename T1::symmetry,
                           typename T1::index_list, typename T1::args_list>& t1,
    const TensorExpression<T2, typename T2::type, typename T2::symmetry,
                           typename T2::index_list, tmpl::list<Args2...>>& t2) {
  return tenex::Divide<T1, T2, Args2...>(~t1, ~t2);
}

/// @{
/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the quotient of a tensor
/// expression over a number
///
/// \note The implementation instead uses the operation, `t * (1.0 / number)`
///
/// \param t the tensor expression operand of the quotient
/// \param number the numeric operand of the quotient
/// \return the tensor expression representing the quotient of the tensor
/// expression over the number
template <typename T, typename N, Requires<std::is_arithmetic_v<N>> = nullptr>
SPECTRE_ALWAYS_INLINE auto operator/(
    const TensorExpression<T, typename T::type, typename T::symmetry,
                           typename T::index_list, typename T::args_list>& t,
    const N number) {
  return t * tenex::NumberAsExpression(1.0 / number);
}
template <typename T, typename N>
SPECTRE_ALWAYS_INLINE auto operator/(
    const TensorExpression<T, typename T::type, typename T::symmetry,
                           typename T::index_list, typename T::args_list>& t,
    const std::complex<N>& number) {
  return t * tenex::NumberAsExpression(1.0 / number);
}
/// @}

/// @{
/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the quotient of a number
/// over a tensor expression that evaluates to a rank 0 tensor
///
/// \param number the numeric numerator of the quotient
/// \param t the tensor expression denominator of the quotient
/// \return the tensor expression representing the quotient of the number over
/// the tensor expression
template <typename T, typename N, Requires<std::is_arithmetic_v<N>> = nullptr>
SPECTRE_ALWAYS_INLINE auto operator/(
    const N number,
    const TensorExpression<T, typename T::type, typename T::symmetry,
                           typename T::index_list, typename T::args_list>& t) {
  return tenex::NumberAsExpression(number) / t;
}
template <typename T, typename N>
SPECTRE_ALWAYS_INLINE auto operator/(
    const std::complex<N>& number,
    const TensorExpression<T, typename T::type, typename T::symmetry,
                           typename T::index_list, typename T::args_list>& t) {
  return tenex::NumberAsExpression(number) / t;
}
/// @}
