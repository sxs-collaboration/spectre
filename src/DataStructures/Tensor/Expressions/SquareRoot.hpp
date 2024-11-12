// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

#include "DataStructures/Tensor/Expressions/NumberAsExpression.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Expressions/TimeIndex.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

namespace tenex {
/// \ingroup TensorExpressionsGroup
/// \brief Defines the tensor expression representing the square root of a
/// tensor expression that evaluates to a rank 0 tensor
///
/// \details The expression can have a non-zero number of indices as long as
/// all indices are concrete time indices, as this represents a rank 0 tensor.
///
/// For details on aliases and members defined in this class, as well as general
/// `TensorExpression` terminology used in its members' documentation, see
/// documentation for `TensorExpression`.
///
/// \tparam T the type of the tensor expression of which to take the square
/// root
/// \tparam Args the TensorIndexs of the expression
template <typename T, typename... Args>
struct SquareRoot
    : public TensorExpression<SquareRoot<T, Args...>, typename T::type,
                              typename T::symmetry, typename T::index_list,
                              tmpl::list<Args...>> {
  static_assert(
      (... and tt::is_time_index<Args>::value),
      "Can only take the square root of a tensor expression that evaluates to "
      "a rank 0 tensor.");

  // === Index properties ===
  /// The type of the data being stored in the result of the expression
  using type = typename T::type;
  /// The ::Symmetry of the result of the expression
  using symmetry = typename T::symmetry;
  /// The list of \ref SpacetimeIndex "TensorIndexType"s of the result of the
  /// expression
  using index_list = typename T::index_list;
  /// The list of generic `TensorIndex`s of the result of the expression
  using args_list = tmpl::list<Args...>;
  /// The number of tensor indices in the result of the expression
  static constexpr auto num_tensor_indices = sizeof...(Args);

  // === Expression subtree properties ===
  /// The number of arithmetic tensor operations done in the subtree for the
  /// left operand
  static constexpr size_t num_ops_left_child = T::num_ops_subtree;
  /// The number of arithmetic tensor operations done in the subtree for the
  /// right operand. This is 0 because this expression represents a unary
  /// operation.
  static constexpr size_t num_ops_right_child = 0;
  /// The total number of arithmetic tensor operations done in this expression's
  /// whole subtree
  static constexpr size_t num_ops_subtree = num_ops_left_child + 1;
  /// The height of this expression's node in the expression tree relative to
  /// the closest `TensorAsExpression` leaf in its subtree
  static constexpr size_t height_relative_to_closest_tensor_leaf_in_subtree =
      T::height_relative_to_closest_tensor_leaf_in_subtree !=
              std::numeric_limits<size_t>::max()
          ? T::height_relative_to_closest_tensor_leaf_in_subtree + 1
          : T::height_relative_to_closest_tensor_leaf_in_subtree;

  // === Properties for splitting up subexpressions along the primary path ===
  // These definitions only have meaning if this expression actually ends up
  // being along the primary path that is taken when evaluating the whole tree.
  // See documentation for `TensorExpression` for more details.
  /// If on the primary path, whether or not the expression is an ending point
  /// of a leg
  static constexpr bool is_primary_end = T::is_primary_start;
  /// If on the primary path, this is the remaining number of arithmetic tensor
  /// operations that need to be done in the subtree of the child along the
  /// primary path, given that we will have already computed the whole subtree
  /// at the next lowest leg's starting point.
  static constexpr size_t num_ops_to_evaluate_primary_left_child =
      is_primary_end ? 0 : T::num_ops_to_evaluate_primary_subtree;
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
  /// If on the primary path, whether or not the expression's child along the
  /// primary path is a subtree that contains a starting point of a leg along
  /// the primary path
  static constexpr bool primary_child_subtree_contains_primary_start =
      T::primary_subtree_contains_primary_start;
  /// If on the primary path, whether or not this subtree contains a starting
  /// point of a leg along the primary path
  static constexpr bool primary_subtree_contains_primary_start =
      is_primary_start or primary_child_subtree_contains_primary_start;

  SquareRoot(T t) : t_(std::move(t)) {}
  ~SquareRoot() override = default;

  /// \brief Assert that the LHS tensor of the equation does not also appear in
  /// this expression's subtree
  template <typename LhsTensor>
  SPECTRE_ALWAYS_INLINE void assert_lhs_tensor_not_in_rhs_expression(
      const gsl::not_null<LhsTensor*> lhs_tensor) const {
    if constexpr (not std::is_base_of_v<MarkAsNumberAsExpression, T>) {
      t_.assert_lhs_tensor_not_in_rhs_expression(lhs_tensor);
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
    if constexpr (not std::is_base_of_v<MarkAsNumberAsExpression, T>) {
      t_.template assert_lhs_tensorindices_same_in_rhs<LhsTensorIndices>(
          lhs_tensor);
    }
  }

  /// \brief Get the size of a component from a `Tensor` in this expression's
  /// subtree of the RHS `TensorExpression`
  ///
  /// \return the size of a component from a `Tensor` in this expression's
  /// subtree of the RHS `TensorExpression`
  SPECTRE_ALWAYS_INLINE size_t get_rhs_tensor_component_size() const {
    return t_.get_rhs_tensor_component_size();
  }

  /// \brief Returns the square root of the component of the tensor evaluated
  /// from the contained tensor expression
  ///
  /// \details
  /// SquareRoot only supports tensor expressions that evaluate to a rank 0
  /// Tensor. This is why `multi_index` is always an array of size 0.
  ///
  /// \param multi_index the multi-index of the component of which to take the
  /// square root
  /// \return the square root of the component of the tensor evaluated from the
  /// contained tensor expression
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    return sqrt(t_.get(multi_index));
  }

  /// \brief Returns the square root of the component of the tensor evaluated
  /// from the contained tensor expression
  ///
  /// \details
  /// SquareRoot only supports tensor expressions that evaluate to a rank 0
  /// Tensor. This is why `multi_index` is always an array of size 0.
  ///
  /// This function differs from `get` in that it takes into account whether we
  /// have already computed part of the result component at a lower subtree.
  /// In recursively computing this square root, the current result component
  /// will be substituted in for the most recent (highest) subtree below it that
  /// has already been evaluated.
  ///
  /// \param result_component the LHS tensor component to evaluate
  /// \param multi_index the multi-index of the component of which to take the
  /// square root
  /// \return the square root of the component of the tensor evaluated from the
  /// contained tensor expression
  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE decltype(auto) get_primary(
      const ResultType& result_component,
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    if constexpr (is_primary_end) {
      (void)multi_index;
      // We've already computed the whole child subtree on the primary path, so
      // just return the square root of the current result component
      return sqrt(result_component);
    } else {
      // We haven't yet evaluated the whole subtree for this expression, so
      // return the square root of this expression's subtree
      return sqrt(t_.get_primary(result_component, multi_index));
    }
  }

  /// \brief Successively evaluate the LHS Tensor's result component at each
  /// leg in this expression's subtree
  ///
  /// \details
  /// This function takes into account whether we have already computed part of
  /// the result component at a lower subtree. In recursively computing this
  /// square root, the current result component will be substituted in for the
  /// most recent (highest) subtree below it that has already been evaluated.
  ///
  /// \param result_component the LHS tensor component to evaluate
  /// \param multi_index the multi-index of the component of the result tensor
  /// to evaluate
  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE void evaluate_primary_subtree(
      ResultType& result_component,
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    if constexpr (primary_child_subtree_contains_primary_start) {
      // The primary child's subtree contains at least one leg, so recurse down
      // and evaluate that first
      t_.evaluate_primary_subtree(result_component, multi_index);
    }

    if constexpr (is_primary_start) {
      // We want to evaluate the subtree for this expression
      result_component = get_primary(result_component, multi_index);
    }
  }

 private:
  /// Operand expression
  T t_;
};
}  // namespace tenex

/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the square root of a
/// tensor expression that evaluates to a rank 0 tensor
///
/// \details
/// `t` must be an expression that, when evaluated, would be a rank 0 tensor.
/// For example, if `R` and `S` are Tensors, here is a non-exhaustive list of
/// some of the acceptable forms that `t` could take:
/// - `R()`
/// - `R(ti::A, ti::a)`
/// - `(R(ti::A, ti::B) * S(ti::a, ti::b))`
/// - `R(ti::t, ti::t) + 1.0`
///
/// \param t the tensor expression of which to take the square root
template <typename T, typename X, typename Symm, typename IndexList,
          typename... Args>
SPECTRE_ALWAYS_INLINE auto sqrt(
    const TensorExpression<T, X, Symm, IndexList, tmpl::list<Args...>>& t) {
  return tenex::SquareRoot<T, Args...>(~t);
}
