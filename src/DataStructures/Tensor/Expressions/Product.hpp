// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines ET for tensor inner and outer products

#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

#include "DataStructures/Tensor/Expressions/Contract.hpp"
#include "DataStructures/Tensor/Expressions/DataTypeSupport.hpp"
#include "DataStructures/Tensor/Expressions/NumberAsExpression.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace tenex {
namespace detail {
template <typename T1, typename T2, typename SymmList1 = typename T1::symmetry,
          typename SymmList2 = typename T2::symmetry>
struct OuterProductType;

template <typename T1, typename T2, template <typename...> class SymmList1,
          typename... Symm1, template <typename...> class SymmList2,
          typename... Symm2>
struct OuterProductType<T1, T2, SymmList1<Symm1...>, SymmList2<Symm2...>> {
  using type =
      typename get_binop_datatype<typename T1::type, typename T2::type>::type;
  using symmetry =
      Symmetry<(Symm1::value + sizeof...(Symm2))..., Symm2::value...>;
  using index_list =
      tmpl::append<typename T1::index_list, typename T2::index_list>;
  using tensorindex_list =
      tmpl::append<typename T1::args_list, typename T2::args_list>;
};
}  // namespace detail

/// \ingroup TensorExpressionsGroup
/// \brief Defines the tensor expression representing the outer product of two
/// tensor expressions
///
/// \details
/// For details on aliases and members defined in this class, as well as general
/// `TensorExpression` terminology used in its members' documentation, see
/// documentation for `TensorExpression`.
///
/// \tparam T1 the left operand expression of the outer product expression
/// \tparam T2 the right operand expression of the outer product expression
template <typename T1, typename T2,
          typename IndexList1 = typename T1::index_list,
          typename IndexList2 = typename T2::index_list,
          typename ArgsList1 = typename T1::args_list,
          typename ArgsList2 = typename T2::args_list>
struct OuterProduct;

template <typename T1, typename T2, template <typename...> class IndexList1,
          typename... Indices1, template <typename...> class IndexList2,
          typename... Indices2, template <typename...> class ArgsList1,
          typename... Args1, template <typename...> class ArgsList2,
          typename... Args2>
struct OuterProduct<T1, T2, IndexList1<Indices1...>, IndexList2<Indices2...>,
                    ArgsList1<Args1...>, ArgsList2<Args2...>>
    : public TensorExpression<
          OuterProduct<T1, T2>, typename detail::OuterProductType<T1, T2>::type,
          typename detail::OuterProductType<T1, T2>::symmetry,
          typename detail::OuterProductType<T1, T2>::index_list,
          typename detail::OuterProductType<T1, T2>::tensorindex_list> {
  static_assert(
      detail::tensorexpression_binop_datatypes_are_supported_v<T1, T2>,
      "Cannot multiply the given TensorExpressions with the given data types. "
      "This can occur from e.g. trying to multiply a Tensor with data type "
      "double and a Tensor with data type DataVector.");
  // === Index properties ===
  /// The type of the data being stored in the result of the expression
  using type = typename detail::OuterProductType<T1, T2>::type;
  /// The ::Symmetry of the result of the expression
  using symmetry = typename detail::OuterProductType<T1, T2>::symmetry;
  /// The list of \ref SpacetimeIndex "TensorIndexType"s of the result of the
  /// expression
  using index_list = typename detail::OuterProductType<T1, T2>::index_list;
  /// The list of generic `TensorIndex`s of the result of the
  /// expression
  using args_list = typename detail::OuterProductType<T1, T2>::tensorindex_list;
  /// The number of tensor indices in the result of the expression
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;
  /// The number of tensor indices in the left operand expression
  static constexpr auto op1_num_tensor_indices =
      tmpl::size<typename T1::index_list>::value;
  /// The number of tensor indices in the right operand expression
  static constexpr auto op2_num_tensor_indices =
      num_tensor_indices - op1_num_tensor_indices;

  // === Expression subtree properties ===
  /// The number of arithmetic tensor operations done in the subtree for the
  /// left operand
  static constexpr size_t num_ops_left_child = T1::num_ops_subtree;
  /// The number of arithmetic tensor operations done in the subtree for the
  /// right operand
  static constexpr size_t num_ops_right_child = T2::num_ops_subtree;
  // This helps ensure we are minimizing breadth in the overall tree
  static_assert(num_ops_left_child >= num_ops_right_child,
                "The left operand should be a subtree with equal or more "
                "tensor operations than the right operand's subtree.");
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

  OuterProduct(T1 t1, T2 t2) : t1_(std::move(t1)), t2_(std::move(t2)) {}
  ~OuterProduct() override = default;

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
      t1_.template assert_lhs_tensorindices_same_in_rhs<LhsTensorIndices>(
          lhs_tensor);
    }
    if constexpr (not std::is_base_of_v<MarkAsNumberAsExpression, T2>) {
      t2_.template assert_lhs_tensorindices_same_in_rhs<LhsTensorIndices>(
          lhs_tensor);
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

  /// \brief Return the first operand's multi-index given the outer product's
  /// multi-index
  ///
  /// \param result_multi_index the multi-index of the component of the outer
  /// product tensor
  /// \return the first operand's multi-index
  constexpr SPECTRE_ALWAYS_INLINE std::array<size_t, op1_num_tensor_indices>
  get_op1_multi_index(
      const std::array<size_t, num_tensor_indices>& result_multi_index) const {
    std::array<size_t, op1_num_tensor_indices> op1_multi_index{};
    for (size_t i = 0; i < op1_num_tensor_indices; i++) {
      gsl::at(op1_multi_index, i) = gsl::at(result_multi_index, i);
    }
    return op1_multi_index;
  }

  /// \brief Return the second operand's multi-index given the outer product's
  /// multi-index
  ///
  /// \param result_multi_index the multi-index of the component of the outer
  /// product tensor
  /// \return the second operand's multi-index
  constexpr SPECTRE_ALWAYS_INLINE std::array<size_t, op2_num_tensor_indices>
  get_op2_multi_index(
      const std::array<size_t, num_tensor_indices>& result_multi_index) const {
    std::array<size_t, op2_num_tensor_indices> op2_multi_index{};
    for (size_t i = 0; i < op2_num_tensor_indices; i++) {
      gsl::at(op2_multi_index, i) =
          gsl::at(result_multi_index, op1_num_tensor_indices + i);
    }
    return op2_multi_index;
  }

  /// \brief Return the value of the component of the outer product tensor at a
  /// given multi-index
  ///
  /// \details
  /// This function takes the multi-index of some component of the resultant
  /// outer product to compute. The function first computes the multi-indices of
  /// the pair of components in the two operand expressions, then multiplies the
  /// values at these multi-indices to obtain the value of the resultant outer
  /// product component. For example, say we are evaluating
  /// \f$L_abc = R_{b} * S_{ca}\f$. Let `result_multi_index == {0, 1, 2}`, which
  /// refers to the component \f$L_{012}\f$, the component we wish to compute.
  /// This function will compute the multi-indices of the operands that
  /// correspond to \f$R_{1}\f$ and \f$S_{20}\f$, retrieve their values, and
  /// return their product.
  ///
  /// \param result_multi_index the multi-index of the component of the outer
  /// product tensor to retrieve
  /// \return the value of the component at `result_multi_index` in the outer
  /// product tensor
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<size_t, num_tensor_indices>& result_multi_index) const {
    return t1_.get(get_op1_multi_index(result_multi_index)) *
           t2_.get(get_op2_multi_index(result_multi_index));
  }

  /// \brief Return the product of the components at the given multi-indices of
  /// the left and right operands
  ///
  /// \details
  /// This function differs from `get` in that it takes into account whether we
  /// have already computed part of the result component at a lower subtree.
  /// In recursively computing this product, the current result component will
  /// be substituted in for the most recent (highest) subtree below it that has
  /// already been evaluated.
  ///
  /// \param result_component the LHS tensor component to evaluate
  /// \param op1_multi_index the multi-index of the component of the first
  /// operand of the product to retrieve
  /// \param op2_multi_index the multi-index of the component of the second
  /// operand of the product to retrieve
  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE decltype(auto) get_primary(
      const ResultType& result_component,
      const std::array<size_t, op1_num_tensor_indices>& op1_multi_index,
      const std::array<size_t, op2_num_tensor_indices>& op2_multi_index) const {
    if constexpr (is_primary_end) {
      (void)op1_multi_index;
      // We've already computed the whole child subtree on the primary path, so
      // just return the product of the current result component and the result
      // of the other child's subtree
      return result_component * t2_.get(op2_multi_index);
    } else {
      // We haven't yet evaluated the whole subtree for this expression, so
      // return the product of the results of the two operands' subtrees
      return t1_.get_primary(result_component, op1_multi_index) *
             t2_.get(op2_multi_index);
    }
  }

  /// \brief Return the value of the component of the outer product tensor at a
  /// given multi-index
  ///
  /// \details
  /// This function differs from `get` in that it takes into account whether we
  /// have already computed part of the result component at a lower subtree.
  /// In recursively computing this product, the current result component will
  /// be substituted in for the most recent (highest) subtree below it that has
  /// already been evaluated.
  ///
  /// \param result_component the LHS tensor component to evaluate
  /// \param result_multi_index the multi-index of the component of the outer
  /// product tensor to retrieve
  /// \return the value of the component at `result_multi_index` in the outer
  /// product tensor
  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE decltype(auto) get_primary(
      const ResultType& result_component,
      const std::array<size_t, num_tensor_indices>& result_multi_index) const {
    return get_primary(result_component,
                       get_op1_multi_index(result_multi_index),
                       get_op2_multi_index(result_multi_index));
  }

  /// \brief Evaluate the LHS Tensor's result component at this subtree by
  /// evaluating the two operand's subtrees separately and multiplying
  ///
  /// \details
  /// This function takes into account whether we have already computed part of
  /// the result component at a lower subtree. In recursively computing this
  /// product, the current result component will be substituted in for the most
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
  /// \param op1_multi_index the multi-index of the component of the first
  /// operand of the product to evaluate
  /// \param op2_multi_index the multi-index of the component of the second
  /// operand of the product to evaluate
  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE void evaluate_primary_children(
      ResultType& result_component,
      const std::array<size_t, op1_num_tensor_indices>& op1_multi_index,
      const std::array<size_t, op2_num_tensor_indices>& op2_multi_index) const {
    if constexpr (is_primary_end) {
      (void)op1_multi_index;
      // We've already computed the whole child subtree on the primary path, so
      // just multiply the current result by the result of the other child's
      // subtree
      result_component *= t2_.get(op2_multi_index);
    } else {
      // We haven't yet evaluated the whole subtree of the primary child, so
      // first assign the result component to be the result of computing the
      // primary child's subtree
      result_component = t1_.get_primary(result_component, op1_multi_index);
      // Now that the primary child's subtree has been computed, multiply the
      // current result by the result of evaluating the other child's subtree
      result_component *= t2_.get(op2_multi_index);
    }
  }

  /// \brief Successively evaluate the LHS Tensor's result component at each
  /// leg in this expression's subtree
  ///
  /// \details
  /// This function takes into account whether we have already computed part of
  /// the result component at a lower subtree. In recursively computing this
  /// product, the current result component will be substituted in for the most
  /// recent (highest) subtree below it that has already been evaluated.
  ///
  /// \param result_component the LHS tensor component to evaluate
  /// \param result_multi_index the multi-index of the component of the outer
  /// product tensor to evaluate
  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE void evaluate_primary_subtree(
      ResultType& result_component,
      const std::array<size_t, num_tensor_indices>& result_multi_index) const {
    const std::array<size_t, op1_num_tensor_indices> op1_multi_index =
        get_op1_multi_index(result_multi_index);
    if constexpr (primary_child_subtree_contains_primary_start) {
      // The primary child's subtree contains at least one leg, so recurse down
      // and evaluate that first
      t1_.evaluate_primary_subtree(result_component, op1_multi_index);
    }

    if constexpr (is_primary_start) {
      // We want to evaluate the subtree for this expression
      if constexpr (evaluate_children_separately) {
        // Evaluate operand's subtrees separately
        evaluate_primary_children(result_component, op1_multi_index,
                                  get_op2_multi_index(result_multi_index));
      } else {
        // Evaluate whole subtree as one expression
        result_component = get_primary(result_component, op1_multi_index,
                                       get_op2_multi_index(result_multi_index));
      }
    }
  }

 private:
  /// Left operand expression
  T1 t1_;
  /// Right operand expression
  T2 t2_;
};
}  // namespace tenex

/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the product of two tensor
/// expressions
///
/// \details
/// If the two operands have N pairs of indices that need to be contracted, the
/// returned expression will be an `OuterProduct` expression nested inside N
/// `TensorContract` expressions. This represents computing the inner product
/// of the outer product of the two operands. If the operands do not have any
/// indices to be contracted, the returned expression will be an `OuterProduct`.
///
/// The two arguments are expressions that contain the two operands of the
/// product, where the types of the operands are `T1` and `T2`.
///
/// \tparam T1 the derived TensorExpression type of the first operand of the
/// product
/// \tparam T2 the derived TensorExpression type of the second operand of the
/// product
/// \tparam ArgsList1 the TensorIndexs of the first operand
/// \tparam ArgsList2 the TensorIndexs of the second operand
/// \param t1 first operand expression of the product
/// \param t2 the second operand expression of the product
/// \return the tensor expression representing the product of two tensor
/// expressions
template <typename T1, typename T2, typename ArgsList1, typename ArgsList2>
SPECTRE_ALWAYS_INLINE auto operator*(
    const TensorExpression<T1, typename T1::type, typename T1::symmetry,
                           typename T1::index_list, ArgsList1>& t1,
    const TensorExpression<T2, typename T2::type, typename T2::symmetry,
                           typename T2::index_list, ArgsList2>& t2) {
  if constexpr (T1::num_ops_subtree >= T2::num_ops_subtree) {
    return tenex::contract(tenex::OuterProduct<T1, T2>(~t1, ~t2));
  } else {
    return tenex::contract(tenex::OuterProduct<T2, T1>(~t2, ~t1));
  }
}

/// @{
/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the product of a tensor
/// expression and a number
///
/// \param t the tensor expression operand of the product
/// \param number the numeric operand of the product
/// \return the tensor expression representing the product of the tensor
/// expression and the number
template <typename T, typename N, Requires<std::is_arithmetic_v<N>> = nullptr>
SPECTRE_ALWAYS_INLINE auto operator*(
    const TensorExpression<T, typename T::type, typename T::symmetry,
                           typename T::index_list, typename T::args_list>& t,
    const N number) {
  return t * tenex::NumberAsExpression(number);
}
template <typename T, typename N, Requires<std::is_arithmetic_v<N>> = nullptr>
SPECTRE_ALWAYS_INLINE auto operator*(
    const N number,
    const TensorExpression<T, typename T::type, typename T::symmetry,
                           typename T::index_list, typename T::args_list>& t) {
  return t * tenex::NumberAsExpression(number);
}
template <typename T, typename N>
SPECTRE_ALWAYS_INLINE auto operator*(
    const TensorExpression<T, typename T::type, typename T::symmetry,
                           typename T::index_list, typename T::args_list>& t,
    const std::complex<N>& number) {
  return t * tenex::NumberAsExpression(number);
}
template <typename T, typename N>
SPECTRE_ALWAYS_INLINE auto operator*(
    const std::complex<N>& number,
    const TensorExpression<T, typename T::type, typename T::symmetry,
                           typename T::index_list, typename T::args_list>& t) {
  return t * tenex::NumberAsExpression(number);
}
/// @}
