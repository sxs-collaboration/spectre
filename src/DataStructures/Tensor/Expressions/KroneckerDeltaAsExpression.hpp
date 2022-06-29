// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines expressions that represent the Kronecker delta

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace tenex {
using KroneckerDeltaFrame = Frame::NoFrame;

/// \ingroup TensorExpressionsGroup
/// \brief Defines an expression representing a
/// \ref KroneckerDelta "Kronecker delta"
///
/// \tparam K the type of the \ref KroneckerDelta "Kronecker delta" being
/// represented
/// \tparam TensorIndex1 the first \ref TensorIndex "generic index"
/// \tparam TensorIndex2 the second \ref TensorIndex "generic index"
template <typename K, typename TensorIndex1, typename TensorIndex2>
struct KroneckerDeltaAsExpression
    : public TensorExpression<
          KroneckerDeltaAsExpression<K, TensorIndex1, TensorIndex2>, double,
          Symmetry<2, 1>,
          index_list<Tensor_detail::TensorIndexType<
                         K::dim, TensorIndex1::valence, KroneckerDeltaFrame,
                         (TensorIndex1::is_spacetime ? IndexType::Spacetime
                                                     : IndexType::Spatial)>,
                     Tensor_detail::TensorIndexType<
                         K::dim, TensorIndex2::valence, KroneckerDeltaFrame,
                         (TensorIndex1::is_spacetime ? IndexType::Spacetime
                                                     : IndexType::Spatial)>>,
          tmpl::list<TensorIndex1, TensorIndex2>>,
      MarkAsDoubleValuedLeafExpression,
      MarkAsNonTensorLeafExpression {
  // === Index properties ===
  /// The type of the data being stored in the result of the expression
  using type = double;
  /// The ::Symmetry of the result of the expression
  using symmetry = Symmetry<2, 1>;
  /// The list of \ref SpacetimeIndex "TensorIndexType"s of the result of the
  /// expression
  using index_list =
      tmpl::list<Tensor_detail::TensorIndexType<
                     K::dim, TensorIndex1::valence, KroneckerDeltaFrame,
                     (TensorIndex1::is_spacetime ? IndexType::Spacetime
                                                 : IndexType::Spatial)>,
                 Tensor_detail::TensorIndexType<
                     K::dim, TensorIndex2::valence, KroneckerDeltaFrame,
                     (TensorIndex1::is_spacetime ? IndexType::Spacetime
                                                 : IndexType::Spatial)>>;
  /// The list of generic `TensorIndex`s of the result of the expression
  using args_list = tmpl::list<TensorIndex1, TensorIndex2>;
  /// The number of tensor indices in the result of the expression
  static constexpr size_t num_tensor_indices = 2;

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
  /// If on the primary path, whether or not the expression is a starting point
  /// of a leg
  static constexpr bool is_primary_start = false;
  /// If on the primary path, whether or not the expression's child along the
  /// primary path is a subtree that contains a starting point of a leg along
  /// the primary path. This is always falls because this expression is a leaf.
  static constexpr bool primary_child_subtree_contains_primary_start = false;
  /// If on the primary path, whether or not this subtree contains a starting
  /// point of a leg along the primary path
  static constexpr bool primary_subtree_contains_primary_start =
      is_primary_start;

  /// \brief Construct a `KroneckerDeltaAsExpression` from a `KroneckerDelta`
  ///
  /// \param k the `KroneckerDelta` to represent as a `TensorExpression`
  KroneckerDeltaAsExpression(const K& k) : k_(&k) {}
  ~KroneckerDeltaAsExpression() override = default;

  // This expression does not represent a `Tensor`, nor does it have any
  // children, so we should never need to assert that the LHS `Tensor` is not
  // equal to the `KroneckerDelta` stored by this expression
  template <typename LhsTensor>
  void assert_lhs_tensor_not_in_rhs_expression(
      const gsl::not_null<LhsTensor*>) const = delete;
  // This expression does not represent a `Tensor`, nor does it have any
  // children, so we should never need to assert that the LHS `Tensor` is not
  // equal to the `KroneckerDelta` stored by this expression
  template <typename LhsTensorIndices, typename LhsTensor>
  void assert_lhs_tensorindices_same_in_rhs(
      const gsl::not_null<LhsTensor*> lhs_tensor) const = delete;

  /// \brief Returns the value of the contained Kronecker delta's multi-index
  ///
  /// \param multi_index the multi-index of the component to retrieve
  /// \return the value of the component at `multi_index` in the Kronecker delta
  constexpr SPECTRE_ALWAYS_INLINE double get(
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    if (gsl::at(multi_index, 0) != gsl::at(multi_index, 1)) {
      return 0.0;
    } else {
      return 1.0;
    }
  }

  /// \brief Returns the value of the contained Kronecker delta's multi-index
  ///
  /// \param multi_index the multi-index of the component to retrieve
  /// \return the value of the component at `multi_index` in the Kronecker delta
  template <typename ResultType>
  constexpr SPECTRE_ALWAYS_INLINE double get_primary(
      const ResultType& /*result_component*/,
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    if (gsl::at(multi_index, 0) != gsl::at(multi_index, 1)) {
      return 0.0;
    } else {
      return 1.0;
    }
  }

  // This expression is a leaf but does not store any information related to the
  // sizing of the tensor components because it does not represent an actual
  // `Tensor`. Therefore, this expression should never be used to initialize a
  // LHS result tensor component. We would run into trouble if e.g. the tensor
  // components in the equations are `DataVector`s, but then we initialize a LHS
  // component using a Kronecker delta element stored in this leaf expression on
  // the primary path.
  template <typename ResultType>
  void evaluate_primary_subtree(
      ResultType&,
      const std::array<size_t, num_tensor_indices>&) const = delete;

 private:
  /// Kronecker delta represented by this expression
  const K* k_ = nullptr;
};
}  // namespace tenex
