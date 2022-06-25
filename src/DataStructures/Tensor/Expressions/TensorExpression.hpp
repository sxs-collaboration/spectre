// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines base class for all tensor expressions

#pragma once

#include <limits>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup TensorExpressionsGroup
/// \brief Marks a class as being a TensorExpression
///
/// \details
/// The empty base class provides a simple means for checking if a type is a
/// TensorExpression.
struct Expression {};

/// @{
/// \ingroup TensorExpressionsGroup
/// \brief The base class all tensor expression implementations derive from
///
/// \details
/// ## Tensor equation construction
/// Each derived `TensorExpression` class should be thought of as an expression
/// tree that represents some operation done on or between tensor expressions.
/// Arithmetic operators and other mathematical functions of interest
/// (e.g. `sqrt`) have overloads defined that accept `TensorExpression`s and
/// return a new `TensorExpression` representing the result tensor of such an
/// operation. In this way, an equation written with `TensorExpression`s will
/// generate an expression tree where the internal and leaf nodes are instances
/// of the derived `TensorExpression` classes. For example, `tenex::AddSub`
/// defines an internal node for handling the addition and subtraction
/// operations between tensors expressions, while `tenex::TensorAsExpression`
/// defines a leaf node that represents a single `Tensor` that appears in the
/// equation.
///
/// ## Tensor equation evaluation
/// The overall tree for an equation and the order in which we traverse the tree
/// define the order of operations done to compute the resulting LHS `Tensor`.
/// The evaluation is done by `tenex::evaluate`, which traverses the whole tree
/// once for each unique LHS component in order to evaluate the full LHS
/// `Tensor`. There are two different traversals currently implemented that are
/// chosen from, depending on the tensor equation being evaluated:
/// 1. **Evaluate the whole tree as one expression** using in-order traversal.
/// This is like generating and solving a one-liner of the whole equation.
/// 2. **Split up the tree into subexpressions** that are each evaluated with
/// in-order traversal to successively "accumulate" a LHS result component of
/// the equation. This is like splitting the equation up and solving pieces of
/// it at a time with multiple lines of assignments/updates (see details below).
///
/// ## Equation splitting details
/// Splitting up the tree and evaluating subexpressions is beneficial when we
/// believe it to lead to a better runtime than if we were to compute the whole
/// expression as a one-liner. One important use case is when the `Tensor`s in
/// the equation hold components whose data type is `DataVector`. From
/// benchmarking, it was found that the runtime of `DataVector` expressions
/// scales poorly as we increase the number of operations. For example, for an
/// inner product with 256 sums of products, instead of adding 256 `DataVector`
/// products in one line (e.g. `result = A*B + C*D + E*F + ...;`), it's much
/// faster to, say, set the result to be the sum of the first 8 products, then
/// `+=` the next 8, and so forth. This is what is meant by "accumulating" the
/// LHS result tensor, and what the `TensorExpression` splitting emulates. Note
/// that while 8 is the number used in this example, the exact optimal number of
/// operations will be hardware-dependent, but probably not something we need to
/// really worry about fine-tuning. However, a ballpark estimate for a "good"
/// number of operations may vary greatly depending on the data type of the
/// components (e.g. `double` vs. `DataVector`), which is something important
/// to at least coarsely tune.
///
/// ### How the tree is split up
/// Let's define the **primary path** to be the path in the tree going from the
/// root node to the leftmost leaf. The overall tree contains subtrees
/// represented by different `TensorExpression`s in the equation. Certain
/// subtrees are marked as the starting and/or ending points of these "pieces"
/// of the equation. Let's define a **leg** to be a "segment" along the primary
/// path delineated by a starting and ending expression subtree. These
/// delineations are made where we decide there are enough operations in a
/// subtree that it would be wise to split at that point. What is considered to
/// be "enough" operations is specialized based on the data type held by the
/// `Tensor`s in the expression (see `tenex::max_num_ops_in_sub_expression`).
///
/// ### How a split tree is traversed and evaluated
/// We recurse down the primary path, visiting each expression subtree until we
/// reach the start of the lowest leg, then initialize the LHS result component
/// we're wanting to compute to be the result of this lowest expression. Then,
/// we recurse back up to the expression subtree that is starting point of the
/// leg "above" it and compute that subtree. This time, however, when
/// recursively evaluating this higher subtree, we substitute in the current LHS
/// result for that lower subtree that we have already computed. This is
/// repeated as we "climb up" the primary path to successively accumulate the
/// result component.
///
/// **Note:** The primary path is currently implemented as the path specified
/// above, but there's no reason it couldn't be reimplemented to be a different
/// path. The idea with the current implementation is to select a path from root
/// to leaf that is long so we have more flexibility in splitting, should we
/// want to. When evaluating, we *could* implement the traversal to take a
/// different path, but currently, derived `TensorExpression`s that represent
/// commutative binary operations are instantiated with the larger subtree being
/// the left child and the smaller subtree being the right child. By
/// constructing it this way, we elongate the leftmost path, which will allow
/// for increased splitting.
///
/// ## Requirements for derived `TensorExpression` classes
/// Each derived `TensorExpression` class must define the following aliases and
/// members:
/// - `private` variables that store its operands' derived `TensorExpression`s.
/// We make these non-`const` to allow for move construction.
/// - Constructor that initializes the above `private` operand members
/// - alias `type`: The data type of the data being stored in the result of the
/// expression, e.g. `double`, `DataVector`
/// - alias `symmetry`: The ::Symmetry of the result of the expression
/// - alias `index_list`: The list of \ref SpacetimeIndex "TensorIndexType"s of
/// the result of the expression
/// - alias `args_list`: The list of generic `TensorIndex`s of the result of the
/// expression
/// - variable `static constexpr size_t num_tensor_indices`: The number of
/// tensor indices in the result of the expression
/// - variable `static constexpr size_t num_ops_left_child`: The number of
/// arithmetic tensor operations done in the subtree for the expression's left
/// operand. If the expression represents a unary operation, their only child is
/// considered the left child. If the expression is a leaf node, then this value
/// should be set to 0 since retrieving a value at the leaf involves 0
/// arithmetic tensor operations.
/// - variable `static constexpr size_t num_ops_right_child`: The number of
/// arithmetic tensor operations done in the expression's right operand. If the
/// expression represents a unary operation or is leaf node, this should be set
/// to 0 because there is no right child.
/// - variable `static constexpr size_t num_ops_subtree`: The number of
/// arithmetic tensor operations done in the subtree represented by the
/// expression. For `AddSub`, for example, this is
/// `num_ops_left_child + num_ops_right_child + 1`, the sum of the number of
/// operations in each operand's subtrees plus one for the operation done for
/// the expression, itself.
/// - function `decltype(auto) get(const std::array<size_t, num_tensor_indices>&
/// result_multi_index) const`: Accepts a multi-index for the result tensor
/// represented by the expression and returns the computed result of the
/// expression at that multi-index. This should call the operands' `get`
/// functions in order to recursively compute the result of the expression.
/// - function template
/// `template <typename LhsTensor> void assert_lhs_tensor_not_in_rhs_expression(
/// const gsl::not_null<LhsTensor*> lhs_tensor) const`: Asserts that the LHS
/// `Tensor` we're computing does not also appear in the RHS `TensorExpression`.
/// We define this because if a tree is split up, then the LHS `Tensor` will
/// generally not be computed correctly because the LHS components will be
/// updated as we traverse the split tree.
/// - function template
/// \code
/// template <typename LhsTensorIndices, typename LhsTensor>
/// void assert_lhs_tensorindices_same_in_rhs(
///     const gsl::not_null<LhsTensor*> lhs_tensor) const;
/// \endcode
/// Asserts that any instance of the LHS `Tensor` in the RHS `TensorExpression`
/// uses the same generic index order that the LHS uses. We define this because
/// if a tree is not split up, it's safe to use the LHS `Tensor` on the RHS if
/// the generic index order is the same. In these cases, `tenex::update` should
/// be used instead of `tenex::evaluate`. See the documentation for
/// `tenex::update` for more details and `tenex::detail::evaluate_impl` for why
/// this is safe to do.
///
/// Each derived `TensorExpression` class must also define the following
/// members, which have real meaning for the expression *only* if it ends up
/// belonging to the primary path of the tree that is traversed:
/// - variable `static constexpr bool is_primary_start`: If on the primary path,
/// whether or not the expression is a starting point of a leg. This is true
/// when there are enough operations to warrant splitting (see
/// `tenex::max_num_ops_in_sub_expression`).
/// - variable `static constexpr bool is_primary_end`: If on the primary path,
/// whether or not the expression is an ending point of a leg. This is true when
/// the expression's child along the primary path is a starting point of a leg.
/// - variable `static constexpr size_t num_ops_to_evaluate_primary_left_child`:
/// If on the primary path, this is the remaining number of arithmetic tensor
/// operations that need to be done in the subtree of the child along the
/// primary path, given that we will have already computed the whole subtree at
/// the next lowest leg's starting point.
/// - variable
/// `static constexpr size_t num_ops_to_evaluate_primary_right_child`:
/// If on the primary path, this is the remaining number of arithmetic tensor
/// operations that need to be done in the right operand's subtree. Because
/// the branches off of the primary path currently are not split up in any way,
/// this currently should simply be equal to `num_ops_right_child`. If logic is
/// added to split up these branches, logic will need to be added to compute
/// this remaining number of operations in the right subtree.
/// - variable `static constexpr size_t num_ops_to_evaluate_primary_subtree`:
/// If on the primary path, this is the remaining number of arithmetic tensor
/// operations that need to be done for this expression's subtree, given that we
/// will have already computed the subtree at the next lowest leg's starting
/// point. For example, for `tenex::AddSub`, this is just
/// `num_ops_to_evaluate_primary_left_child +
/// num_ops_to_evaluate_primary_right_child + 1` (the extra 1 for the `+` or `-`
/// operation itself).
/// - variable
/// `static constexpr bool primary_child_subtree_contains_primary_start`:
/// If on the primary path, whether or not the expression's child along the
/// primary path is a subtree that contains a starting point of a leg along the
/// primary path. In other words, whether or not there is a split on the primary
/// path lower than this expression. When evaluating a split tree, this is
/// useful because it tells us we need to keep recursing down to a lower leg and
/// evaluate that lower subtree first before evaluating the current subtree.
/// - variable `static constexpr bool primary_subtree_contains_primary_start`:
/// If on the primary path, whether or not this subtree contains a starting
/// point of a leg along the primary path. In other words, whether or not there
/// is a split on the primary path at this expression or beneath it.
/// - function `decltype(auto) get_primary(const type& result_component,
/// const std::array<size_t, num_tensor_indices>& result_multi_index) const`:
/// This is similar to the required `get` function described above, but this
/// should be used when the tree is split up. The main difference with this
/// function is that it takes the current result component (that we're
/// computing) as an argument, and when we hit the starting point of the next
/// lowest leg on the primary path when recursively evaluating the current leg,
/// we substitute in the current LHS result for the subtree that we have already
/// computed. This function should call `get_primary` on the child on the
/// primary path and `get` on the other child, if one exists.
/// - function `void evaluate_primary_subtree(type& result_component,
/// const std::array<size_t, num_tensor_indices>& result_multi_index) const`:
/// This should first recursively evaluate the legs beneath it on the primary
/// path, then if the expression itself is the start of a leg, it should
/// evaluate this leg by calling the expression's own `get_primary` to compute
/// it and update the result component being accumulated. `tenex::evaluate`
/// should call this function on the root node for the whole tree if there is
/// determined to be any splits in the tree.
///
/// ## Current advice for improving and extending `TensorExpression`s
/// - Derived `TensorExpression` classes (or the overloads that produce them)
/// should include `static_assert`s for ensuring mathematical correctness
/// wherever reasonable
/// - Minimize breadth in the tree where possible because benchmarking inner
/// products has shown that increased tree breadth can cause slower runtimes.
/// In addition, more breadth means a decreased ability to split up the tree
/// along the primary path.
/// - Minimize the number of multi-index transformations that need to be done
/// when evaluating the tree. For some operations like addition, the associated
/// multi-indices of the two operands needs to be computed from the multi-index
/// of the result, which may involve reordering and/or shifting the values of
/// the result index. It's good to minimize the number of these kinds of
/// transformations from result to operand multi-index where we can.
/// - Unless the implementation of Tensor_detail::Structure changes, it's not
/// advised for the derived `TensorExpression` classes to have anything that
/// would instantiate the Tensor_detail::Structure of the tensor that would
/// result from the expression. This is really only a problem when the result of
/// the expression would be a tensor with many components, because the compile
/// time of the mapping between storage indices and multi-indices within
/// Tensor_detail::Structure scales very poorly with the number of components.
/// It's important to keep in mind that while SpECTRE currently only supports
/// creating `Tensor`s up to rank 4, there is nothing preventing the represented
/// result tensor of a expression being higher rank, e.g.
/// `R(ti_j, ti_b, ti_A) * (S(ti_d, ti_a, ti_B, ti_C) * T(ti_J, ti_k, ti_l))`
/// contains an intermediate outer product expression
/// `S(ti_d, ti_a, ti_B, ti_C) * T(ti_J, ti_k, ti_l)` that represents a rank 7
/// tensor, even though a rank 7 `Tensor` is never instantiated. Having the
/// outer product expression instantiate the Tensor_detail::Structure of this
/// intermediate result currently leads to an unreasonable compile time.
///
/// \tparam Derived the derived class needed for
/// [CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)
/// \tparam DataType the type of the data being stored in the `Tensor`s
/// \tparam Symm the ::Symmetry of the Derived class
/// \tparam IndexList the list of \ref SpacetimeIndex "TensorIndexType"s
/// \tparam Args typelist of the tensor indices, e.g. types of `ti::a` and
/// `ti::b` in `F(ti::a, ti::b)`
/// \cond HIDDEN_SYMBOLS
template <typename Derived, typename DataType, typename Symm,
          typename IndexList, typename Args = tmpl::list<>,
          typename ReducedArgs = tmpl::list<>>
struct TensorExpression;
/// \endcond

template <typename Derived, typename DataType, typename Symm,
          typename... Indices, template <typename...> class ArgsList,
          typename... Args>
struct TensorExpression<Derived, DataType, Symm, tmpl::list<Indices...>,
                        ArgsList<Args...>> : public Expression {
  static_assert(sizeof...(Args) == 0 or sizeof...(Args) == sizeof...(Indices),
                "the number of Tensor indices must match the number of "
                "components specified in an expression.");
  /// The type of the data being stored in the `Tensor`s
  using type = DataType;
  /// The ::Symmetry of the `Derived` class
  using symmetry = Symm;
  /// The list of \ref SpacetimeIndex "TensorIndexType"s
  using index_list = tmpl::list<Indices...>;
  /// Typelist of the tensor indices, e.g. types of `ti_a` and `ti_b`
  /// in `F(ti_a, ti_b)`
  using args_list = ArgsList<Args...>;
  /// The number of tensor indices of the `Derived` class
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;

  virtual ~TensorExpression() = 0;

  /// @{
  /// Derived is casted down to the derived class. This is enabled by the
  /// [CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)
  ///
  /// \returns const TensorExpression<Derived, DataType, Symm, IndexList,
  /// ArgsList<Args...>>&
  SPECTRE_ALWAYS_INLINE const auto& operator~() const {
    return static_cast<const Derived&>(*this);
  }
  /// @}
};

template <typename Derived, typename DataType, typename Symm,
          typename... Indices, template <typename...> class ArgsList,
          typename... Args>
TensorExpression<Derived, DataType, Symm, tmpl::list<Indices...>,
                 ArgsList<Args...>>::~TensorExpression() = default;
/// @}

namespace tenex {
namespace detail {
/// @{
/// \brief The maximum number of arithmetic tensor operations allowed in a
/// `TensorExpression` subtree before having it be a splitting point in the
/// overall RHS expression, according to the data type held by the `Tensor`s in
/// the expression
///
/// \details
/// To enable splitting for `TensorExpression`s with a different data type,
/// define a new variable below like `max_num_ops_in_datavector_sub_expression`
/// for your data type, then update the control flow in
/// `max_num_ops_in_sub_expression_helper`.
///
/// Before defining a max operations cap for some data type, the change should
/// first be justified by benchmarking many different tensor expressions before
/// and after introducing the new cap. The optimal cap will likely be
/// hardware-dependent, so fine-tuning this would ideally involve benchmarking
/// on each hardware architecture and then controling the value based on the
/// hardware.
///
/// The current value set for when the data type is `DataVector` was benchmarked
/// by compiling with clang-10 Release and running on Intel(R) Xeon(R)
/// CPU E5-2630 v4 @ 2.20GHz.
static constexpr size_t max_num_ops_in_datavector_sub_expression = 8;
/// @}

/// \brief Helper struct for getting the maximum number of arithmetic tensor
/// operations allowed in a `TensorExpression` subtree before having it be a
/// splitting point in the overall RHS expression, according to the `DataType`
/// held by the `Tensor`s in the expression
///
/// \tparam DataType the type of the data being stored in the `Tensor`s in the
/// `TensorExpression`
template <typename DataType>
struct max_num_ops_in_sub_expression_helper {
  // Splitting is only enabled for expressions when DataType == DataVector
  // because benchmarking has shown it to be beneficial. To enable splitting
  // for other data types, define a new static variable like
  // `max_num_ops_in_datavector_sub_expression` for the data type of interest,
  // then update the control flow below
  static constexpr size_t value = std::is_same_v<DataType, DataVector>
                                      ? max_num_ops_in_datavector_sub_expression
                                      // effectively, no splitting
                                      : std::numeric_limits<size_t>::max();
};

/// \brief Get maximum number of arithmetic tensor operations allowed in a
/// `TensorExpression` subtree before having it be a splitting point in the
/// overall RHS expression, according to the `DataType` held by the `Tensor`s in
/// the expression
template <typename DataType>
inline constexpr size_t max_num_ops_in_sub_expression =
    max_num_ops_in_sub_expression_helper<DataType>::value;
}  // namespace detail
}  // namespace tenex
