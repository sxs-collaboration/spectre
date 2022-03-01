// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

namespace tenex {
/// \ingroup TensorExpressionsGroup
/// \brief Defines an expression representing a `double`
///
/// \details
/// For details on aliases and members defined in this class, as well as general
/// `TensorExpression` terminology used in its members' documentation, see
/// documentation for `TensorExpression`.
struct NumberAsExpression
    : public TensorExpression<NumberAsExpression, double, tmpl::list<>,
                              tmpl::list<>, tmpl::list<>> {
  // === Index properties ===
  /// The type of the data being stored in the result of the expression
  using type = double;
  /// The list of \ref SpacetimeIndex "TensorIndexType"s of the result of the
  /// expression
  using symmetry = tmpl::list<>;
  /// The list of \ref SpacetimeIndex "TensorIndexType"s of the result of the
  /// expression
  using index_list = tmpl::list<>;
  /// The list of generic `TensorIndex`s of the result of the expression
  using args_list = tmpl::list<>;
  /// The number of tensor indices in the result of the expression
  static constexpr auto num_tensor_indices = 0;

  // === Arithmetic tensor operations properties ===
  /// The number of arithmetic tensor operations done in the subtree for the
  /// left operand, which is 0 because this is a leaf expression
  static constexpr size_t num_ops_left_child = 0;
  /// The number of arithmetic tensor operations done in the subtree for the
  /// right operand, which is 0 because this is a leaf expression
  static constexpr size_t num_ops_right_child = 0;
  /// The total number of arithmetic tensor operations done in this expression's
  /// whole subtree, which is 0 because this is a leaf expression
  static constexpr size_t num_ops_subtree = 0;

  // === Properties for splitting up subexpressions along the primary path ===
  // These definitions only have meaning if this expression actually ends up
  // being along the primary path that is taken when evaluating the whole tree.
  // See documentation for `TensorExpression` for more details.
  /// If on the primary path, whether or not the expression is an ending point
  /// of a leg
  static constexpr bool is_primary_end = true;
  /// If on the primary path, this is the remaining number of arithmetic tensor
  /// operations that need to be done in the subtree of the child along the
  /// primary path, given that we will have already computed the whole subtree
  /// at the next lowest leg's starting point. This is just 0 because this
  /// expression is a leaf.
  static constexpr size_t num_ops_to_evaluate_primary_left_child = 0;
  /// If on the primary path, this is the remaining number of arithmetic tensor
  /// operations that need to be done in the right operand's subtree. This is
  /// just 0 because this expression is a leaf.
  static constexpr size_t num_ops_to_evaluate_primary_right_child = 0;
  /// If on the primary path, this is the remaining number of arithmetic tensor
  /// operations that need to be done for this expression's subtree, given that
  /// we will have already computed the subtree at the next lowest leg's
  /// starting point. This is just 0 because this expression is a leaf.
  static constexpr size_t num_ops_to_evaluate_primary_subtree = 0;
  /// \brief If on the primary path, whether or not the expression is a starting
  /// point of a leg
  ///
  /// \details
  /// Note: it's especially important for `NumberAsExpression` to define this as
  /// `false` because if we have a case where this kind of an expression is the
  /// leaf of the first leg being evaluated in the overall tree, we don't want
  /// to use a `double` to initialize/size one of our LHS tensor's components,
  /// because things would break if the LHS tensor components are supposed to
  /// be e.g. a `DataVector` with a specific size.
  static constexpr bool is_primary_start = false;
  /// If on the primary path, whether or not the expression's child along the
  /// primary path is a subtree that contains a starting point of a leg along
  /// the primary path. This is always falls because this expression is a leaf.
  static constexpr bool primary_child_subtree_contains_primary_start = false;
  /// If on the primary path, whether or not this subtree contains a starting
  /// point of a leg along the primary path
  static constexpr bool primary_subtree_contains_primary_start =
      is_primary_start;

  NumberAsExpression(const double number) : number_(number) {}
  ~NumberAsExpression() override = default;

  // This expression does not represent a tensor, nor does it have any children,
  // so we should never need to assert that the LHS `Tensor` is not equal to the
  // `double` stored by this expression
  template <typename LhsTensor>
  void assert_lhs_tensor_not_in_rhs_expression(
      const gsl::not_null<LhsTensor*>) const = delete;
  // This expression does not represent a tensor, nor does it have any children,
  // so we should never need to assert that instances of the LHS Tensor in this
  // expression's subtree have the same generic index order
  template <typename LhsTensorIndices, typename LhsTensor>
  void assert_lhs_tensorindices_same_in_rhs(
      const gsl::not_null<LhsTensor*> lhs_tensor) const = delete;

  /// \brief Returns the number represented by the expression
  ///
  /// \return the number represented by this expression
  SPECTRE_ALWAYS_INLINE double get(
      const std::array<size_t, num_tensor_indices>& /*multi_index*/) const {
    return number_;
  }

  /// \brief Returns the number represented by the expression
  ///
  /// \return the number represented by this expression
  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE double get_primary(
      const ResultType& /*result_component*/,
      const std::array<size_t, num_tensor_indices>& /*multi_index*/) const {
    return number_;
  }

  // This expression is a leaf but does not store any information related to the
  // sizing of the tensor components, because it does not represent an actual
  // expression with tensors. Therefore, this expression should never be used to
  // initialize a LHS result tensor component. We would run into trouble if e.g.
  // the tensor components in the equations are `DataVector`s, but then we
  // initialize a LHS component using the `double` stored in this leaf
  // expression on the primary path.
  template <typename ResultType>
  void evaluate_primary_subtree(
      ResultType&,
      const std::array<size_t, num_tensor_indices>&) const = delete;

 private:
  /// Number represented by this expression
  double number_;
};
}  // namespace tenex
