// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines Expression Templates for contracting tensor indices on a single
/// Tensor

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

#include "DataStructures/Tensor/Expressions/NumberAsExpression.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/Expressions/TimeIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \ingroup TensorExpressionsGroup
 * Holds all possible TensorExpressions currently implemented
 */
namespace tenex {
namespace detail {
template <typename I1, typename I2>
using indices_contractible = std::bool_constant<
    I1::ul != I2::ul and
    std::is_same_v<typename I1::Frame, typename I2::Frame> and
    ((I1::index_type == I2::index_type and I1::dim == I2::dim) or
     // If one index is spacetime and the other is spatial, the indices can
     // be contracted if they have the same number of spatial dimensions
     (I1::index_type == IndexType::Spacetime and I1::dim == I2::dim + 1) or
     (I2::index_type == IndexType::Spacetime and I1::dim + 1 == I2::dim))>;

template <size_t NumUncontractedIndices>
constexpr size_t get_num_contracted_index_pairs(
    const std::array<size_t, NumUncontractedIndices>&
        uncontracted_tensor_index_values) {
  size_t count = 0;
  for (size_t i = 0; i < NumUncontractedIndices; i++) {
    const size_t current_value = gsl::at(uncontracted_tensor_index_values, i);
    // Concrete time indices are not contracted
    if (not detail::is_time_index_value(current_value)) {
      const size_t opposite_value_to_find =
          get_tensorindex_value_with_opposite_valence(current_value);
      for (size_t j = i + 1; j < NumUncontractedIndices; j++) {
        if (opposite_value_to_find ==
            gsl::at(uncontracted_tensor_index_values, j)) {
          // We found both the lower and upper version of a generic index in the
          // list of generic indices, so we return this pair's positions
          count++;
        }
      }
    }
  }

  return count;
}

/// \brief Computes the mapping from the positions of indices in the resultant
/// contracted tensor to their positions in the operand uncontracted tensor, as
/// well as the positions of the index pairs in the operand uncontracted tensor
/// that we wish to contract
///
/// \details
/// Both quantities returned are computed in and returned from the same
/// function so as to not repeat overlapping necessary work that would be in two
/// separate functions
///
/// \tparam NumContractedIndexPairs the number of pairs of indices that will be
/// contracted
/// \tparam NumUncontractedIndices the number of indices in the operand
/// expression that we wish to contract
/// \param uncontracted_tensor_index_values the values of the `TensorIndex`s
/// used to generically represent the uncontracted operand expression
/// \return the mapping from the positions of indices in the resultant
/// contracted tensor to their positions in the operand uncontracted tensor, as
/// well as the positions of the index pairs in the operand uncontracted tensor
/// that we wish to contract
template <size_t NumContractedIndexPairs, size_t NumUncontractedIndices>
constexpr std::pair<
    std::array<size_t, NumUncontractedIndices - NumContractedIndexPairs * 2>,
    std::array<std::pair<size_t, size_t>, NumContractedIndexPairs>>
get_index_transformation_and_contracted_pair_positions(
    const std::array<size_t, NumUncontractedIndices>&
        uncontracted_tensor_index_values) {
  static_assert(NumUncontractedIndices >= 2,
                "There should be at least 2 indices");
  // Positions of indices in the result tensor (ones that are not contracted)
  // mapped to their locations in the uncontracted operand expression
  std::array<size_t, NumUncontractedIndices - NumContractedIndexPairs * 2>
      index_transformation{};
  // Positions of contracted index pairs in the uncontracted operand expression
  std::array<std::pair<size_t, size_t>, NumContractedIndexPairs>
      contracted_index_pair_positions{};

  // Marks whether or not we have already paired an index in the uncontracted
  // operand expression with an index to contract it with
  std::array<bool, NumUncontractedIndices> index_mapping_set =
      make_array<NumUncontractedIndices, bool>(false);

  // Index of `contracted_index_pair_positions` that we're currently assigning
  size_t contracted_map_index_to_assign = 0;
  // Index of `index_transformation` that we're currently assigning
  size_t not_contracted_map_index_to_assign =
      NumUncontractedIndices - NumContractedIndexPairs * 2 - 1;
  // Iteration is performed backwards here for the reasons below, but note that
  // no benchmarking has been done to confirm the backwards iteration order
  // makes a meaningful improvement in runtime vs. iterating forward:
  //
  // Here, we iterate backwards to find the "rightmost" contracted indices and
  // proceed leftwards to find the other contracted index pairs so that the
  // "rightmost" pairs in the expression to contract appear first in the list of
  // contracted index pairs (`contracted_index_pair_positions`). Then, when we
  // later iterate over all of the multi-indices to sum, each time we go to grab
  // the next multi-index to sum and we need to compute what that next
  // multi-index is, we can choose to increment the concrete values of the
  // rightmost index pair.
  //
  // This has not been benchmarked to confirm, but the thought with making this
  // choice to order the contracted pairs from right to left is that this may
  // help with spatial locality for caching when we are contracting a single
  // tensor (`TensorAsExpression`) that is non-symmetric. For example, let's say
  // we are contracting `R(ti::A, ti::B, ti::a, ti::b)` and the current
  // multi-index we just accessed (one of the components to sum) is
  // `{0, 0, 0, 0}`, representing \f$R^{00}{}_{00}\f$. To find a next
  // multi-index to sum, we can simply increase one of the pairs' concrete
  // values by 1, e.g. the next multi-index to access could be `{0, 1, 0, 1}`
  // for \f$R^{01}{}_{01}\f$ or `{1, 0, 1, 0}` \f$R^{10}{}_{10}\f$. In this
  // case, the idea is that choosing to increment the concrete values of the
  // rightmost pair could provide better spatial locality, as `{0, 1, 0, 1}` is
  // closer in memory to `{0, 0, 0, 0}` than `{1, 0, 1, 0}` is. It's important
  // to note that this, of course, depends on the implementation of
  // `Tensor_detail::Structure` - specifically, the order in which the
  // components are laid out in memory.
  //
  // Note: the loop terminates when underflow causes `i` to wrap back around to
  // the maximum `size_t` value. If we use the condition `i > 0`, we miss the
  // final iteration, and if we use `i >= 0`, we never terminate because `i`
  // is always positive.
  for (size_t i = NumUncontractedIndices - 1; i < NumUncontractedIndices; i--) {
    if (not gsl::at(index_mapping_set, i)) {
      const size_t current_value = gsl::at(uncontracted_tensor_index_values, i);
      // Concrete time indices are not contracted
      if (not detail::is_time_index_value(current_value)) {
        const size_t opposite_value_to_find =
            get_tensorindex_value_with_opposite_valence(current_value);
        for (size_t j = i - 1; j < NumUncontractedIndices; j--) {
          if (opposite_value_to_find ==
              gsl::at(uncontracted_tensor_index_values, j)) {
            // We found both the lower and upper version of a generic index in
            // the list of generic indices, pair them up
            gsl::at(contracted_index_pair_positions,
                    contracted_map_index_to_assign)
                .first = i;
            gsl::at(contracted_index_pair_positions,
                    contracted_map_index_to_assign)
                .second = j;
            contracted_map_index_to_assign++;
            // Mark that we've found contraction partners for these two indices
            gsl::at(index_mapping_set, i) = true;
            gsl::at(index_mapping_set, j) = true;
            break;
          }
        }
      }
      if (not gsl::at(index_mapping_set, i)) {
        // If we haven't assigned this index to a partner, it is not an index
        // that is contracted, so record its position mapping from contracted to
        // uncontracted tensor indices
        gsl::at(index_transformation, not_contracted_map_index_to_assign) = i;
        not_contracted_map_index_to_assign--;
      }
    }
  }

  return std::pair{index_transformation, contracted_index_pair_positions};
}

/// \brief Computes type information for the tensor expression that results from
/// a contraction, as well as information internally useful for carrying out the
/// contraction
///
/// \tparam UncontractedTensorExpression the operand uncontracted
/// `TensorExpression` being contracted
/// \tparam DataType the data type of the `Tensor` components
/// \tparam UncontractedSymm the ::Symmetry of the operand uncontracted
/// `TensorExpression`
/// \tparam UncontractedIndexList the list of
/// \ref SpacetimeIndex "TensorIndexType"s of the operand uncontracted
/// `TensorExpression`
/// \tparam UncontractedTensorIndexList the list of generic `TensorIndex`s used
/// for the the operand uncontracted `TensorExpression`
/// \tparam NumContractedIndices the number of indices in the resultant tensor
/// after contracting
/// \tparam NumIndexPairsToContract the number of pairs of indices that will be
/// contracted
template <typename UncontractedTensorExpression, typename DataType,
          typename UncontractedSymm, typename UncontractedIndexList,
          typename UncontractedTensorIndexList, size_t NumContractedIndices,
          size_t NumIndexPairsToContract,
          typename ContractedIndexSequence =
              std::make_index_sequence<NumContractedIndices>,
          typename IndexPairsToContractSequence =
              std::make_index_sequence<NumIndexPairsToContract>>
struct ContractedType;

template <typename UncontractedTensorExpression, typename DataType,
          template <typename...> class UncontractedSymmList,
          typename... UncontractedSymm,
          template <typename...> class UncontractedIndexList,
          typename... UncontractedIndices,
          template <typename...> class UncontractedTensorIndexList,
          typename... UncontractedTensorIndices, size_t NumContractedIndices,
          size_t NumIndexPairsToContract, size_t... ContractedInts,
          size_t... IndexPairsToContractInts>
struct ContractedType<UncontractedTensorExpression, DataType,
                      UncontractedSymmList<UncontractedSymm...>,
                      UncontractedIndexList<UncontractedIndices...>,
                      UncontractedTensorIndexList<UncontractedTensorIndices...>,
                      NumContractedIndices, NumIndexPairsToContract,
                      std::index_sequence<ContractedInts...>,
                      std::index_sequence<IndexPairsToContractInts...>> {
  static constexpr size_t num_uncontracted_tensor_indices =
      sizeof...(UncontractedTensorIndices);
  static constexpr std::array<size_t, num_uncontracted_tensor_indices>
      uncontracted_tensorindex_values = {{UncontractedTensorIndices::value...}};
  static constexpr size_t num_indices_to_contract =
      num_uncontracted_tensor_indices - NumContractedIndices;
  static constexpr size_t num_contracted_index_pairs =
      num_indices_to_contract / 2;
  // First item in pair:
  // - index transformation: mapping from the positions of indices in the
  // resultant contracted tensor to their positions in the operand
  // uncontracted tensor
  // Second item in pair:
  // contracted index pair positions: positions of the index pairs in the
  // operand uncontracted tensor that we wish to contract
  static constexpr inline std::pair<
      std::array<size_t, NumContractedIndices>,
      std::array<std::pair<size_t, size_t>, num_contracted_index_pairs>>
      index_transformation_and_contracted_pair_positions =
          get_index_transformation_and_contracted_pair_positions<
              num_contracted_index_pairs, num_uncontracted_tensor_indices>(
              uncontracted_tensorindex_values);

  // Make sure it's mathematically legal to perform the requested contraction
  static_assert(((... and
                  (indices_contractible<
                      typename tmpl::at_c<
                          tmpl::list<UncontractedIndices...>,
                          index_transformation_and_contracted_pair_positions
                              .second[IndexPairsToContractInts]
                              .first>,
                      typename tmpl::at_c<
                          tmpl::list<UncontractedIndices...>,
                          index_transformation_and_contracted_pair_positions
                              .second[IndexPairsToContractInts]
                              .second>>::value))),
                "Cannot contract the requested indices.");

  static constexpr inline std::array<IndexType, num_uncontracted_tensor_indices>
      uncontracted_index_types = {{UncontractedIndices::index_type...}};

  // First concrete values of contracted indices to sum. This is to handle
  // cases when we have generic spatial `TensorIndex`s used for spacetime
  // indices, as the first concrete index value to contract will be 1 (first
  // spatial index) instead of 0 (the time index). Contracted index pairs will
  // have different "starting" concrete indices when one index in the pair is a
  // spatial spacetime index and the other is not.
  static constexpr inline std::array<std::pair<size_t, size_t>,
                                     num_contracted_index_pairs>
      contracted_index_first_values = []() {
        std::array<std::pair<size_t, size_t>, num_contracted_index_pairs>
            first_values{};
        for (size_t i = 0; i < num_contracted_index_pairs; i++) {
          // Assign the value for first index in a pair to be the smallest value
          // used in the terms being summed: assign to 1 if we have a spacetime
          // index where a generic spatial index has been used, otherwise assign
          // to 0.
          gsl::at(first_values, i).first = static_cast<size_t>(
              gsl::at(
                  uncontracted_index_types,
                  gsl::at(
                      index_transformation_and_contracted_pair_positions.second,
                      i)
                      .first) == IndexType::Spacetime and
              gsl::at(
                  uncontracted_tensorindex_values,
                  gsl::at(
                      index_transformation_and_contracted_pair_positions.second,
                      i)
                      .first) >= TensorIndex_detail::spatial_sentinel);
          // Assign the value for second index in a pair to be the smallest
          // value used in the terms being summed (assigned with same logic
          // described above for the first index in the pair)
          gsl::at(first_values, i).second = static_cast<size_t>(
              gsl::at(
                  uncontracted_index_types,
                  gsl::at(
                      index_transformation_and_contracted_pair_positions.second,
                      i)
                      .second) == IndexType::Spacetime and
              gsl::at(
                  uncontracted_tensorindex_values,
                  gsl::at(
                      index_transformation_and_contracted_pair_positions.second,
                      i)
                      .second) >= TensorIndex_detail::spatial_sentinel);
        }
        return first_values;
      }();

  static constexpr inline std::array<size_t, num_uncontracted_tensor_indices>
      uncontracted_index_dims = {{UncontractedIndices::dim...}};

  // The number of terms to sum for this expression's contraction
  static constexpr size_t num_terms_summed = []() {
    size_t num_terms =
        gsl::at(
            uncontracted_index_dims,
            gsl::at(index_transformation_and_contracted_pair_positions.second,
                    0)
                .first) -
        gsl::at(contracted_index_first_values, 0).first;
    for (size_t i = 1; i < num_contracted_index_pairs; i++) {
      num_terms *=
          gsl::at(
              uncontracted_index_dims,
              gsl::at(index_transformation_and_contracted_pair_positions.second,
                      i)
                  .first) -
          gsl::at(contracted_index_first_values, i).first;
    }
    return num_terms;
  }();
  static_assert(num_terms_summed > 0,
                "There should be a non-zero number of components to sum in the "
                "contraction.");
  // The ::Symmetry of the result of the contraction
  using symmetry =
      Symmetry<tmpl::at_c<UncontractedSymmList<UncontractedSymm...>,
                          index_transformation_and_contracted_pair_positions
                              .first[ContractedInts]>::value...>;
  // The list of \ref SpacetimeIndex "TensorIndexType"s of the result of the
  // contraction
  using index_list =
      tmpl::list<tmpl::at_c<UncontractedIndexList<UncontractedIndices...>,
                            index_transformation_and_contracted_pair_positions
                                .first[ContractedInts]>...>;
  // The list of generic `TensorIndex`s of the result of the contraction
  using tensorindex_list = tmpl::list<
      tmpl::at_c<UncontractedTensorIndexList<UncontractedTensorIndices...>,
                 index_transformation_and_contracted_pair_positions
                     .first[ContractedInts]>...>;
  // The `TensorExpression` type that results from performing the contraction
  using type = TensorExpression<UncontractedTensorExpression, DataType,
                                symmetry, index_list, tensorindex_list>;
};
}  // namespace detail

/*!
 * \ingroup TensorExpressionsGroup
 */
template <typename T, typename X, typename Symm, typename IndexList,
          typename ArgsList, size_t NumContractedIndices>
struct TensorContract
    : public TensorExpression<
          TensorContract<T, X, Symm, IndexList, ArgsList, NumContractedIndices>,
          X,
          typename detail::ContractedType<
              T, X, Symm, IndexList, ArgsList, NumContractedIndices,
              (tmpl::size<Symm>::value - NumContractedIndices) /
                  2>::type::symmetry,
          typename detail::ContractedType<
              T, X, Symm, IndexList, ArgsList, NumContractedIndices,
              (tmpl::size<Symm>::value - NumContractedIndices) /
                  2>::type::index_list,
          typename detail::ContractedType<
              T, X, Symm, IndexList, ArgsList, NumContractedIndices,
              (tmpl::size<Symm>::value - NumContractedIndices) /
                  2>::type::args_list> {
  /// Stores internally useful information regarding the contraction. See
  /// `detail::ContractedType` for more details
  using contracted_type = typename detail::ContractedType<
      T, X, Symm, IndexList, ArgsList, NumContractedIndices,
      (tmpl::size<Symm>::value - NumContractedIndices) / 2>;
  /// The `TensorExpression` type that results from performing the contraction
  using new_type = typename contracted_type::type;

  // === Index properties ===
  /// The type of the data being stored in the result of the expression
  using type = X;
  /// The ::Symmetry of the result of the expression
  using symmetry = typename new_type::symmetry;
  /// The list of \ref SpacetimeIndex "TensorIndexType"s of the result of the
  /// expression
  using index_list = typename new_type::index_list;
  /// The list of generic `TensorIndex`s of the result of the expression
  using args_list = typename new_type::args_list;
  /// The number of tensor indices in the result of the expression
  static constexpr size_t num_tensor_indices = NumContractedIndices;
  /// The number of tensor indices in the operand expression being contracted
  static constexpr size_t num_uncontracted_tensor_indices =
      tmpl::size<Symm>::value;
  /// The number of tensor indices in the operand expression that will be
  /// contracted
  static constexpr size_t num_indices_to_contract =
      contracted_type::num_indices_to_contract;
  static_assert(num_indices_to_contract > 0,
                "There are no indices to contract that were found.");
  static_assert(num_indices_to_contract % 2 == 0,
                "Cannot contract an odd number of indices.");
  /// The number of tensor index pairs in the operand expression that will be
  /// contracted
  static constexpr size_t num_contracted_index_pairs =
      contracted_type::num_contracted_index_pairs;
  /// Mapping from the positions of indices in the resultant contracted tensor
  /// to their positions in the operand uncontracted tensor
  static constexpr inline std::array<size_t, NumContractedIndices>
      index_transformation =
          contracted_type::index_transformation_and_contracted_pair_positions
              .first;
  /// Positions of the index pairs in the operand uncontracted tensor that we
  /// wish to contract
  static constexpr inline std::array<std::pair<size_t, size_t>,
                                     num_contracted_index_pairs>
      contracted_index_pair_positions =
          contracted_type::index_transformation_and_contracted_pair_positions
              .second;
  /// First concrete values of contracted indices to sum. This is to handle
  /// cases when we have generic spatial `TensorIndex`s used for spacetime
  /// indices, as the first concrete index value to contract will be 1 (first
  /// spatial index) instead of 0 (the time index). Contracted index pairs will
  /// have different "starting" concrete indices when one index in the pair is a
  /// spatial spacetime index and the other is not.
  static constexpr inline std::array<std::pair<size_t, size_t>,
                                     num_contracted_index_pairs>
      contracted_index_first_values =
          contracted_type::contracted_index_first_values;
  /// The dimensions of the indices in the uncontracted operand expression
  static constexpr inline std::array<size_t, num_uncontracted_tensor_indices>
      uncontracted_index_dims = contracted_type::uncontracted_index_dims;
  /// The number of terms to sum for this expression's contraction
  static constexpr size_t num_terms_summed = contracted_type::num_terms_summed;

  // === Expression subtree properties ===
  /// The number of arithmetic tensor operations done in the subtree for the
  /// left operand
  static constexpr size_t num_ops_left_child =
      T::num_ops_subtree * num_terms_summed + num_terms_summed - 1;
  /// The number of arithmetic tensor operations done in the subtree for the
  /// right operand. This is 0 because this expression represents a unary
  /// operation.
  static constexpr size_t num_ops_right_child = 0;
  /// The total number of arithmetic tensor operations done in this expression's
  /// whole subtree
  static constexpr size_t num_ops_subtree = num_ops_left_child;
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
      is_primary_end
          ? num_ops_subtree - T::num_ops_subtree
          : T::num_ops_subtree * (num_terms_summed - 1) +
                T::num_ops_to_evaluate_primary_subtree + num_terms_summed - 1;
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
      num_ops_to_evaluate_primary_right_child;
  /// If on the primary path, whether or not the expression is a starting point
  /// of a leg
  static constexpr bool is_primary_start =
      num_ops_to_evaluate_primary_subtree >
      // Multiply by 2 because each term has a + and * operation, while other
      // arithmetic expression types do one operation
      2 * detail::max_num_ops_in_sub_expression<type>;
  /// If on the primary path, whether or not the expression's child along the
  /// primary path is a subtree that contains a starting point of a leg along
  /// the primary path
  static constexpr bool primary_child_subtree_contains_primary_start =
      T::primary_subtree_contains_primary_start;
  /// If on the primary path, whether or not this subtree contains a starting
  /// point of a leg along the primary path
  static constexpr bool primary_subtree_contains_primary_start =
      is_primary_start or primary_child_subtree_contains_primary_start;
  /// Number of arithmetic tensor operations done in the subtree of the operand
  /// expression being contracted
  static constexpr size_t num_ops_subexpression = T::num_ops_subtree;
  /// In the subtree for this contraction, how many terms we sum together for
  /// each leg of the contraction
  static constexpr size_t leg_length = []() {
    if constexpr (not is_primary_start) {
      // If we're not even stopping at the beginning of the contraction, it's
      // because there weren't enough terms to justify any splitting, so the
      // leg_length is just the total number of terms to sum
      return num_terms_summed;
    } else if constexpr (num_ops_subexpression >=
                         detail::max_num_ops_in_sub_expression<type>) {
      // If the subexpression itself has more than the max # of ops
      return 0;
    } else {
      // Otherwise, find how many terms to sum in each leg
      size_t length = 1;
      while (2 * (length * (num_ops_subexpression + 1) - 1) <=
             detail::max_num_ops_in_sub_expression<type>) {
        length *= 2;
      }
      return length;
    }
  }();
  /// After dividing up the contraction subtree into legs, the number of legs
  /// whose length is equal to `leg_length`
  static constexpr size_t num_full_legs =
      leg_length == 0 ? num_terms_summed : num_terms_summed / leg_length;
  /// After dividing up the contraction subtree into legs of even length, the
  /// number of terms we still have left to sum
  static constexpr size_t last_leg_length =
      leg_length == 0 ? 0 : num_terms_summed % leg_length;
  /// When evaluating along a primary path, whether each term's subtrees should
  /// be evaluated separately. Since `DataVector` expression runtime scales
  /// poorly with increased number of operations, evaluating individual terms'
  /// subtrees separately like this is beneficial when each term, itself,
  /// involves many tensor operations.
  static constexpr bool evaluate_terms_separately = leg_length == 0;

  explicit TensorContract(
      const TensorExpression<T, X, Symm, IndexList, ArgsList>& t)
      : t_(~t) {}
  ~TensorContract() override = default;

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

  /// \brief Return the highest multi-index between the components being summed
  /// in the contraction
  ///
  /// \details
  /// Example:
  /// We have expression `R(ti::A, ti::b, ti::a)` to represent the contraction
  /// \f$L_b = R^{a}{}_{ba}\f$. If the `contracted_multi_index` is `{1}`, which
  /// represents \f$L_1 = R^{a}{}_{1a}\f$, and the dimension of \f$a\f$ is 3,
  /// then we will need to sum the following terms: \f$R^{0}{}_{10}\f$,
  /// \f$R^{1}{}_{11}\f$, and \f$R^{2}{}_{12}\f$. Between the terms being
  /// summed, the multi-index whose values are the largest is
  /// \f$R^{2}{}_{12}\f$, so this function would return `{2, 1, 2}`.
  ///
  /// \param contracted_multi_index the multi-index of a component of the
  /// contracted expression
  /// \return the highest multi-index between the components being summed in
  /// the contraction
  SPECTRE_ALWAYS_INLINE static constexpr std::array<
      size_t, num_uncontracted_tensor_indices>
  get_highest_multi_index_to_sum(
      const std::array<size_t, num_tensor_indices>& contracted_multi_index) {
    // Initialize with placeholders for debugging
    auto highest_multi_index = make_array<num_uncontracted_tensor_indices>(
        std::numeric_limits<size_t>::max());

    // Fill uncontracted indices
    for (size_t i = 0; i < num_tensor_indices; i++) {
      gsl::at(highest_multi_index, gsl::at(index_transformation, i)) =
          gsl::at(contracted_multi_index, i);
    }

    // Fill contracted indices
    for (size_t i = 0; i < num_contracted_index_pairs; i++) {
      const size_t first_index_position_in_pair =
          gsl::at(contracted_index_pair_positions, i).first;
      const size_t second_index_position_in_pair =
          gsl::at(contracted_index_pair_positions, i).second;
      gsl::at(highest_multi_index, first_index_position_in_pair) =
          gsl::at(uncontracted_index_dims, first_index_position_in_pair) - 1;
      gsl::at(highest_multi_index, second_index_position_in_pair) =
          gsl::at(uncontracted_index_dims, second_index_position_in_pair) - 1;
    }

    return highest_multi_index;
  }

  /// \brief Return the lowest multi-index between the components being summed
  /// in the contraction
  ///
  /// \details
  /// Example:
  /// We have expression `R(ti::A, ti::b, ti::a)` to represent the contraction
  /// \f$L_b = R^{a}{}_{ba}\f$. If the `contracted_multi_index` is `{1}`, which
  /// represents \f$L_1 = R^{a}{}_{1a}\f$, and the dimension of \f$a\f$ is 3,
  /// then we will need to sum the following terms: \f$R^{0}{}_{10}\f$,
  /// \f$R^{1}{}_{11}\f$, and \f$R^{2}{}_{12}\f$. Between the terms being
  /// summed, the multi-index whose values are the smallest is
  /// \f$R^{0}{}_{10}\f$, so this function would return `{0, 1, 0}`.
  ///
  /// \param contracted_multi_index the multi-index of a component of the
  /// contracted expression
  /// \return the lowest multi-index between the components being summed in
  /// the contraction
  SPECTRE_ALWAYS_INLINE static constexpr std::array<
      size_t, num_uncontracted_tensor_indices>
  get_lowest_multi_index_to_sum(
      const std::array<size_t, num_tensor_indices>& contracted_multi_index) {
    // Initialize with placeholders for debugging
    auto lowest_multi_index = make_array<num_uncontracted_tensor_indices>(
        std::numeric_limits<size_t>::max());

    // Fill uncontracted indices
    for (size_t i = 0; i < num_tensor_indices; i++) {
      gsl::at(lowest_multi_index, gsl::at(index_transformation, i)) =
          gsl::at(contracted_multi_index, i);
    }

    // Fill contracted indices
    for (size_t i = 0; i < num_contracted_index_pairs; i++) {
      const size_t first_index_position_in_pair =
          gsl::at(contracted_index_pair_positions, i).first;
      const size_t second_index_position_in_pair =
          gsl::at(contracted_index_pair_positions, i).second;
      gsl::at(lowest_multi_index, first_index_position_in_pair) =
          gsl::at(contracted_index_first_values, i).first;
      gsl::at(lowest_multi_index, second_index_position_in_pair) =
          gsl::at(contracted_index_first_values, i).second;
    }

    return lowest_multi_index;
  }

  /// \brief Given the multi-index of one term being summed in the contraction,
  /// return the next highest multi-index of a component being summed
  ///
  /// \details
  /// What is meant by "next highest" is implementation defined, but generally
  /// means, of the components being summed, return the multi-index that results
  /// from lowering one of the contracted index pairs' values by one.
  ///
  /// Example:
  /// We have expression `R(ti::A, ti::b, ti::a)` to represent the contraction
  /// \f$L_b = R^{a}{}_{ba}\f$. If we are evaluating \f$L_1 = R^{a}{}_{1a}\f$
  ///  and the dimension of \f$a\f$ is 3, then we will need to sum the following
  /// terms: \f$R^{0}{}_{10}\f$, \f$R^{1}{}_{11}\f$, and \f$R^{2}{}_{12}\f$.
  /// If `uncontracted_multi_index` is `{1, 1, 1}`, then the "next highest"
  /// multi-index is the result of lowering the values of the \f$a\f$ indices by
  /// 1. The component with that resulting multi-index is \f$R^{0}{}_{10}\f$, so
  /// this function would return `{0, 1, 0}`.
  ///
  /// Note: this function should perform the inverse functionality of
  /// `get_next_highest_multi_index_to_sum`. If the implementation of this
  /// function or the other changes what is meant by "next highest" or "next
  /// lowest," the other function should be updated in accordance.
  ///
  /// \param uncontracted_multi_index the multi-index of one of the components
  /// of the uncontracted operand expression to sum
  /// \return the next highest multi-index between the components being summed
  /// in the contraction
  SPECTRE_ALWAYS_INLINE static std::array<size_t,
                                          num_uncontracted_tensor_indices>
  get_next_highest_multi_index_to_sum(
      const std::array<size_t, num_uncontracted_tensor_indices>&
          uncontracted_multi_index) {
    std::array<size_t, num_uncontracted_tensor_indices>
        next_highest_uncontracted_multi_index = uncontracted_multi_index;

    size_t i = 0;
    while (i < num_contracted_index_pairs) {
      // the position of the first index in a pair being contracted
      const size_t current_index_first_position =
          gsl::at(contracted_index_pair_positions, i).first;
      // the position of the second index in a pair being contracted
      const size_t current_index_second_position =
          gsl::at(contracted_index_pair_positions, i).second;
      // of the values being summed over, the lowest concrete value of the first
      // index in the contracted pair
      const size_t current_index_first_first_value =
          gsl::at(contracted_index_first_values, i).first;

      // decrement the current index pair's values
      gsl::at(next_highest_uncontracted_multi_index,
              current_index_first_position)--;
      gsl::at(next_highest_uncontracted_multi_index,
              current_index_second_position)--;

      // If the index values of the index pair being contracted aren't lower
      // than the minimum values included in the summation, then we're done
      // computing this next multi-index
      if (not(gsl::at(next_highest_uncontracted_multi_index,
                      current_index_first_position) <
                  current_index_first_first_value or
              gsl::at(next_highest_uncontracted_multi_index,
                      current_index_first_position) >
                  gsl::at(uncontracted_index_dims,
                          current_index_first_position))) {
        break;
      }
      // Otherwise, we've wrapped around the lowest value being summed over for
      // this index, so we need to set it back to the maximum values being
      // summed and "carry" the decrementing over to the next contracted pair's
      // values
      gsl::at(next_highest_uncontracted_multi_index,
              current_index_first_position) =
          gsl::at(uncontracted_index_dims, current_index_first_position) - 1;
      gsl::at(next_highest_uncontracted_multi_index,
              current_index_second_position) =
          gsl::at(uncontracted_index_dims, current_index_second_position) - 1;

      i++;
    }

    return next_highest_uncontracted_multi_index;
  }

  /// \brief Given the multi-index of one term being summed in the contraction,
  /// return the next lowest multi-index of a component being summed
  ///
  /// \details
  /// What is meant by "next lowest" is implementation defined, but generally
  /// means, of the components being summed, return the multi-index that results
  /// from raising one of the contracted index pairs' values by one.
  ///
  /// Example:
  /// We have expression `R(ti::A, ti::b, ti::a)` to represent the contraction
  /// \f$L_b = R^{a}{}_{ba}\f$. If we are evaluating \f$L_1 = R^{a}{}_{1a}\f$
  ///  and the dimension of \f$a\f$ is 3, then we will need to sum the following
  /// terms: \f$R^{0}{}_{10}\f$, \f$R^{1}{}_{11}\f$, and \f$R^{2}{}_{12}\f$.
  /// If `uncontracted_multi_index` is `{1, 1, 1}`, then the "next lowest"
  /// multi-index is the result of raising the values of the \f$a\f$ indices by
  /// 1. The component with that resulting multi-index is \f$R^{2}{}_{12}\f$, so
  /// this function would return `{2, 1, 2}`.
  ///
  /// Note: this function should perform the inverse functionality of
  /// `get_next_lowest_multi_index_to_sum`. If the implementation of this
  /// function or the other changes what is meant by "next highest" or "next
  /// lowest," the other function should be updated in accordance.
  ///
  /// \param uncontracted_multi_index the multi-index of one of the components
  /// of the uncontracted operand expression to sum
  /// \return the next lowest multi-index between the components being summed in
  /// the contraction
  SPECTRE_ALWAYS_INLINE static std::array<size_t,
                                          num_uncontracted_tensor_indices>
  get_next_lowest_multi_index_to_sum(
      const std::array<size_t, num_uncontracted_tensor_indices>&
          uncontracted_multi_index) {
    std::array<size_t, num_uncontracted_tensor_indices>
        next_lowest_uncontracted_multi_index = uncontracted_multi_index;

    size_t i = 0;
    while (i < num_contracted_index_pairs) {
      // the position of the first index in a pair being contracted
      const size_t current_index_first_position =
          gsl::at(contracted_index_pair_positions, i).first;
      // the position of the second index in a pair being contracted
      const size_t current_index_second_position =
          gsl::at(contracted_index_pair_positions, i).second;

      // increment the current index pair's values
      gsl::at(next_lowest_uncontracted_multi_index,
              current_index_first_position)++;
      gsl::at(next_lowest_uncontracted_multi_index,
              current_index_second_position)++;

      // if the previous index value is > dim, then we've wrapped around
      // and we need to go again
      // If the index values of the index pair being contracted aren't higher
      // than the maximum values included in the summation, then we're done
      // computing this next multi-index
      if (not(gsl::at(next_lowest_uncontracted_multi_index,
                      current_index_first_position) >
              gsl::at(uncontracted_index_dims, current_index_first_position) -
                  1)) {
        break;
      }
      // Otherwise, we've wrapped around the highest value being summed over for
      // this index, so we need to set it back to the minimum values being
      // summed and "carry" the incrementing over to the next contracted pair's
      // values
      gsl::at(next_lowest_uncontracted_multi_index,
              current_index_first_position) =
          gsl::at(contracted_index_first_values, i).first;
      gsl::at(next_lowest_uncontracted_multi_index,
              current_index_second_position) =
          gsl::at(contracted_index_first_values, i).second;

      i++;
    }

    return next_lowest_uncontracted_multi_index;
  }

  /// \brief Computes the value of a component in the resultant contracted
  /// tensor
  ///
  /// \details
  /// The contraction is computed by recursively adding up each component in the
  /// summation, across all index pairs being contracted in the operand
  /// expression. This function is called `Iteration = num_terms_summed` times,
  /// once for each uncontracted tensor component being summed. It should
  /// externally be called for the first time with `Iteration == 0` and
  /// `current_multi_index == <highest multi index to sum>` (see
  /// `get_next_highest_multi_index_to_sum` for details).
  ///
  /// In performing the recursive summation, the recursion is
  /// specifically done "to the left," in that this function returns
  /// `compute_contraction(next index) + get(this_index)` as opposed to
  /// `get(this_index) + compute_contraction`. Benchmarking has shown that
  /// increased breadth in an equation's expression tree can slow down runtime.
  /// By "recursing left" here, we  minimize breadth in the overall tree for an
  /// equation, as both `AddSub` addition and `OuterProduct` (other expressions
  /// with two children) make efforts to make their operands with larger
  /// subtrees be their left operand.
  ///
  /// \tparam Iteration the nth term to sum, where n is between
  /// [0, num_terms_summed)
  /// \param t the expression contained within this contraction expression
  /// \param current_multi_index the multi-index of the uncontracted tensor
  /// component to retrieve
  /// \return the value of a component of the resulant contracted tensor
  template <size_t Iteration>
  SPECTRE_ALWAYS_INLINE static decltype(auto) compute_contraction(
      const T& t, const std::array<size_t, num_uncontracted_tensor_indices>&
                      current_multi_index) {
    if constexpr (Iteration < num_terms_summed - 1) {
      // We have more than one component left to sum
      return compute_contraction<Iteration + 1>(
                 t, get_next_highest_multi_index_to_sum(current_multi_index)) +
             t.get(current_multi_index);
    } else {
      // We only have one final component to sum
      return t.get(current_multi_index);
    }
  }

  /// \brief Return the value of the component of the resultant contracted
  /// tensor at a given multi-index
  ///
  /// \param contracted_multi_index the multi-index of the resultant contracted
  /// tensor component to retrieve
  /// \return the value of the component at `contracted_multi_index` in the
  /// resultant contracted tensor
  decltype(auto) get(const std::array<size_t, num_tensor_indices>&
                         contracted_multi_index) const {
    return compute_contraction<0>(
        t_, get_highest_multi_index_to_sum(contracted_multi_index));
  }

  /// \brief Computes the result of an internal leg of the contraction
  ///
  /// \details
  /// This function differs from `compute_contraction` and
  /// `compute_contraction_primary` in that it only computes one leg of the
  /// whole contraction, as opposed to the whole contraction.
  ///
  /// The leg being summed is defined by the `current_multi_index` and
  /// `Iteration` passed in from the inital external call: consecutive terms
  /// will be summed until the base case `Iteration == 0` is reached.
  ///
  /// \tparam Iteration the nth term in the leg to sum, where n is between
  /// [0, leg_length)
  /// \param t the expression contained within this contraction expression
  /// \param current_multi_index the multi-index of the uncontracted tensor
  /// component to retrieve as part of this leg's summation
  /// \param next_leg_starting_multi_index in the final iteration, the
  /// multi-index to update to be the next leg's starting multi-index
  /// \return the result of summing up the terms in the given leg
  template <size_t Iteration>
  SPECTRE_ALWAYS_INLINE static decltype(auto) compute_contraction_leg(
      const T& t,
      const std::array<size_t, num_uncontracted_tensor_indices>&
          current_multi_index,
      std::array<size_t, num_uncontracted_tensor_indices>&
          next_leg_starting_multi_index) {
    if constexpr (Iteration != 0) {
      // We have more than one component left to sum
      (void)next_leg_starting_multi_index;
      return compute_contraction_leg<Iteration - 1>(
                 t, get_next_highest_multi_index_to_sum(current_multi_index),
                 next_leg_starting_multi_index) +
             t.get(current_multi_index);
    } else {
      // We only have one final component to sum
      next_leg_starting_multi_index =
          get_next_highest_multi_index_to_sum(current_multi_index);
      return t.get(current_multi_index);
    }
  }

  /// \brief Computes the value of a component in the resultant contracted
  /// tensor
  ///
  /// \details
  /// First see `compute_contraction` for details on basic functionality.
  ///
  /// This function differs from `compute_contraction` in that it takes into
  /// account whether we have already computed part of the result component at a
  /// lower subtree. In recursively computing this contraction, the current
  /// result component will be substituted in for the most recent (highest)
  /// subtree below it that has already been evaluated.
  ///
  /// \tparam Iteration the nth term to sum, where n is between
  /// [0, num_terms_summed)
  /// \param t the expression contained within this contraction expression
  /// \param result_component the LHS tensor component to evaluate
  /// \param current_multi_index the multi-index of the uncontracted tensor
  /// component to retrieve
  /// \return the value of a component of the resulant contracted tensor
  template <size_t Iteration>
  SPECTRE_ALWAYS_INLINE static decltype(auto) compute_contraction_primary(
      const T& t, const type& result_component,
      const std::array<size_t, num_uncontracted_tensor_indices>&
          current_multi_index) {
    if constexpr (is_primary_end) {
      // We've already computed the whole subtree of the term being summed that
      // is at the lowest depth in the tree
      if constexpr (Iteration < num_terms_summed - 1) {
        // We have more than one component left to sum
        return compute_contraction_primary<Iteration + 1>(
                   t, result_component,
                   get_next_highest_multi_index_to_sum(current_multi_index)) +
               t.get(current_multi_index);
      } else {
        // The deepest term in the contraction subtree that is being summed is
        // just our current result, so return it
        return result_component;
      }
    } else {
      // We've haven't yet computed the whole subtree of the term being summed
      // that is at the lowest depth in the tree
      if constexpr (Iteration < num_terms_summed - 1) {
        // We have more than one component left to sum
        return compute_contraction_primary<Iteration + 1>(
                   t, result_component,
                   get_next_highest_multi_index_to_sum(current_multi_index)) +
               t.get(current_multi_index);
      } else {
        // We only have one final component to sum
        return t.get_primary(result_component, current_multi_index);
      }
    }
  }

  /// \brief Return the value of the component of the resultant contracted
  /// tensor at a given multi-index
  ///
  /// \details
  /// This function differs from `get` in that it takes into account whether we
  /// have already computed part of the result component at a lower subtree.
  /// In recursively computing this contraction, the current result component
  /// will be substituted in for the most recent (highest) subtree below it that
  /// has already been evaluated.
  ///
  /// \param result_component the LHS tensor component to evaluate
  /// \param contracted_multi_index the multi-index of the resultant contracted
  /// tensor component to retrieve
  /// \return the value of the component at `contracted_multi_index` in the
  /// resultant contracted tensor
  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE decltype(auto) get_primary(
      const ResultType& result_component,
      const std::array<size_t, num_tensor_indices>& contracted_multi_index)
      const {
    return compute_contraction_primary<0>(
        t_, result_component,
        get_highest_multi_index_to_sum(contracted_multi_index));
  }

  /// \brief Successively evaluate the LHS Tensor's result component at each
  /// leg of summations within the contraction expression
  ///
  /// \details
  /// This function takes into account whether we have already computed part of
  /// the result component at a lower subtree. In recursively computing this
  /// contraction, the current result component will be substituted in for the
  /// most recent (highest) subtree below it that has already been evaluated.
  ///
  /// \param result_component the LHS tensor component to evaluate
  /// \param contracted_multi_index the multi-index of the component of the
  /// contracted result tensor to evaluate
  /// \param lowest_multi_index the lowest multi-index between the components
  /// being summed in the contraction (see `get_lowest_multi_index_to_sum`)
  SPECTRE_ALWAYS_INLINE void evaluate_primary_contraction(
      type& result_component,
      const std::array<size_t, num_tensor_indices>& contracted_multi_index,
      const std::array<size_t, num_uncontracted_tensor_indices>&
          lowest_multi_index) const {
    if constexpr (not is_primary_end) {
      // We need to first evaluate the subtree of the term being summed that
      // is deepest in the tree
      result_component = t_.get_primary(result_component, lowest_multi_index);
    }

    if constexpr (evaluate_terms_separately) {
      // Case 1: Evaluate all of the remaining terms, one TERM at a time
      (void)contracted_multi_index;
      std::array<size_t, num_uncontracted_tensor_indices> current_multi_index =
          lowest_multi_index;
      for (size_t i = 1; i < num_terms_summed; i++) {
        const std::array<size_t, num_uncontracted_tensor_indices>
            next_lowest_multi_index_to_sum =
                get_next_lowest_multi_index_to_sum(current_multi_index);
        result_component += t_.get(next_lowest_multi_index_to_sum);
        current_multi_index = next_lowest_multi_index_to_sum;
      }
    } else {
      // Case 2: Evaluate all of the remaining terms, one LEG at a time
      (void)lowest_multi_index;
      std::array<size_t, num_uncontracted_tensor_indices>
          next_leg_starting_multi_index =
              get_highest_multi_index_to_sum(contracted_multi_index);
      if constexpr (last_leg_length > 0) {
        // Case 2a: We have a remainder of terms that don't make up a full leg
        // length

        // Evaluate all the full-length legs
        for (size_t i = 0; i < num_full_legs; i++) {
          const std::array<size_t, num_uncontracted_tensor_indices>
              current_multi_index = next_leg_starting_multi_index;
          result_component += compute_contraction_leg<leg_length - 1>(
              t_, current_multi_index, next_leg_starting_multi_index);
        }
        if constexpr (last_leg_length > 1) {
          // Get rest of the deepest (partial-length) leg if there are more
          // terms in it than just the one deepest term we already computed
          const std::array<size_t, num_uncontracted_tensor_indices>
              current_multi_index = next_leg_starting_multi_index;
          result_component +=
              // start at last_leg_length - 2 because we already computed one of
              // the terms in this deepest leg (the deepest term)
              compute_contraction_leg<last_leg_length - 2>(
                  t_, current_multi_index, next_leg_starting_multi_index);
        }
      } else {
        // Case 2b: We don't have remaining terms that only make up a
        // partial leg length (i.e. we only have full-length legs)

        // Evaluate all but the deepest leg
        for (size_t i = 1; i < num_full_legs; i++) {
          const std::array<size_t, num_uncontracted_tensor_indices>
              current_multi_index = next_leg_starting_multi_index;

          result_component += compute_contraction_leg<leg_length - 1>(
              t_, current_multi_index, next_leg_starting_multi_index);
        }

        if constexpr (leg_length > 1) {
          // Get rest of the deepest leg if there are more terms in it than
          // just the one deepest term we already computed
          const std::array<size_t, num_uncontracted_tensor_indices>
              current_multi_index = next_leg_starting_multi_index;
          result_component +=
              // start at leg_length - 2 because we already computed one of the
              // terms in this deepest leg (the deepest term)
              compute_contraction_leg<leg_length - 2>(
                  t_, current_multi_index, next_leg_starting_multi_index);
        }
      }
    }
  }

  /// \brief Successively evaluate the LHS Tensor's result component at each
  /// leg in this expression's subtree
  ///
  /// \details
  /// This function takes into account whether we have already computed part of
  /// the result component at a lower subtree. In recursively computing this
  /// contraction, the current result component will be substituted in for the
  /// most recent (highest) subtree below it that has already been evaluated.
  ///
  /// If this contraction expression is the beginning of a leg,
  /// `evaluate_primary_contraction` is called to evaluate each individual
  /// leg of summations within the contraction.
  ///
  /// \param result_component the LHS tensor component to evaluate
  /// \param contracted_multi_index the multi-index of the component of the
  /// contracted result tensor to evaluate
  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE void evaluate_primary_subtree(
      ResultType& result_component,
      const std::array<size_t, num_tensor_indices>& contracted_multi_index)
      const {
    const auto lowest_multi_index_to_sum =
        get_lowest_multi_index_to_sum(contracted_multi_index);
    if constexpr (primary_child_subtree_contains_primary_start) {
      // The primary child's subtree contains at least one leg, so recurse down
      // and evaluate that first. Here, we evaluate the lowest multi-index
      // because, according to `compute_contraction`, the lowest multi-index is
      // the one in the last/leaf/final call to `compute_contraction` (i.e. the
      // multi-index of the final term to sum)
      t_.evaluate_primary_subtree(result_component, lowest_multi_index_to_sum);
    }
    if constexpr (is_primary_start) {
      // We want to evaluate the subtree for this expression, one leg of
      // summations at a time
      evaluate_primary_contraction(result_component, contracted_multi_index,
                                   lowest_multi_index_to_sum);
    }
  }

 private:
  /// Operand expression being contracted
  T t_;
};

template <typename T, typename X, typename Symm, typename IndexList,
          typename... TensorIndices>
SPECTRE_ALWAYS_INLINE static constexpr auto contract(
    const TensorExpression<T, X, Symm, IndexList, tmpl::list<TensorIndices...>>&
        t) {
  // Number of indices in the tensor expression we're wanting to contract
  constexpr size_t num_uncontracted_indices = sizeof...(TensorIndices);
  // The number of index pairs to contract
  constexpr size_t num_contracted_index_pairs =
      detail::get_num_contracted_index_pairs<num_uncontracted_indices>(
          {{TensorIndices::value...}});

  if constexpr (num_contracted_index_pairs == 0) {
    // There aren't any indices to contract, so we just return the input
    return ~t;
  } else {
    // We have at least one pair of indices to contract
    return TensorContract<T, X, Symm, IndexList, tmpl::list<TensorIndices...>,
                          num_uncontracted_indices -
                              (num_contracted_index_pairs * 2)>{t};
  }
}
}  // namespace tenex
