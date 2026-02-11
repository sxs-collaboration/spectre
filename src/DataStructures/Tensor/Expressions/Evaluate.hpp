// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions for evaluating `TensorExpression`s

#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/DataTypeSupport.hpp"
#include "DataStructures/Tensor/Expressions/IndexPropertyCheck.hpp"
#include "DataStructures/Tensor/Expressions/LhsTensorSymmAndIndices.hpp"
#include "DataStructures/Tensor/Expressions/SpatialSpacetimeIndex.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndexTransformation.hpp"
#include "DataStructures/Tensor/Expressions/TimeIndex.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace tenex {
namespace detail {
template <size_t NumIndices>
constexpr bool contains_indices_to_contract(
    const std::array<size_t, NumIndices>& tensorindices) {
  if constexpr (NumIndices < 2) {
    return false;
  } else {
    for (size_t i = 0; i < NumIndices - 1; i++) {
      for (size_t j = i + 1; j < NumIndices; j++) {
        const size_t current_tensorindex = gsl::at(tensorindices, i);
        // Concrete time indices are not contracted
        if ((not is_time_index_value(current_tensorindex)) and
            current_tensorindex == get_tensorindex_value_with_opposite_valence(
                                       gsl::at(tensorindices, j))) {
          return true;
        }
      }
    }
    return false;
  }
}

/// \brief Given the list of the positions of the LHS tensor's spacetime indices
/// where a generic spatial index is used and the list of positions where a
/// concrete time index is used, determine whether or not the component at the
/// given LHS multi-index should be computed
///
/// \details
/// Not all of the LHS tensor's components may need to be computed. Cases when
/// the component at a LHS multi-index should not be not evaluated:
/// - If a generic spatial index is used for a spacetime index on the LHS,
/// the components for which that index's concrete index is the time index
/// should not be computed
/// - If a concrete time index is used for a spacetime index on the LHS, the
/// components for which that index's concrete index is a spatial index should
/// not be computed
///
/// \param lhs_multi_index the multi-index of the LHS tensor to check
/// \param lhs_spatial_spacetime_index_positions the positions of the LHS
/// tensor's spacetime indices where a generic spatial index is used
/// \param lhs_time_index_positions the positions of the LHS tensor's spacetime
/// indices where a concrete time index is used
/// \return Whether or not `lhs_multi_index` is a multi-index of a component of
/// the LHS tensor that should be computed
template <size_t NumLhsIndices, size_t NumLhsSpatialSpacetimeIndices,
          size_t NumLhsConcreteTimeIndices>
constexpr bool is_evaluated_lhs_multi_index(
    const std::array<size_t, NumLhsIndices>& lhs_multi_index,
    const std::array<size_t, NumLhsSpatialSpacetimeIndices>&
        lhs_spatial_spacetime_index_positions,
    const std::array<size_t, NumLhsConcreteTimeIndices>&
        lhs_time_index_positions) {
  for (size_t i = 0; i < lhs_spatial_spacetime_index_positions.size(); i++) {
    if (gsl::at(lhs_multi_index,
                gsl::at(lhs_spatial_spacetime_index_positions, i)) == 0) {
      return false;
    }
  }
  for (size_t i = 0; i < lhs_time_index_positions.size(); i++) {
    if (gsl::at(lhs_multi_index, gsl::at(lhs_time_index_positions, i)) != 0) {
      return false;
    }
  }
  return true;
}

template <typename SymmList>
struct CheckNoLhsAntiSymmetries;

template <template <typename...> class SymmList, typename... Symm>
struct CheckNoLhsAntiSymmetries<SymmList<Symm...>> {
  static constexpr bool value = (... and (Symm::value > 0));
};

/// \brief Given a tensor and its list of tensor indices, return the
/// canonicalized order of the tensor indices according to the tensor's symmetry
///
/// \details
/// The canonical ordering of a `Tensor`'s `TensorIndex`s
/// (e.g. `ti::a`, `ti::b`, ti::c) is relevant to sets of indices that are
/// symmetric. Within each set of symmetric indices, the `TensorIndex` used for
/// each index can be freely reordered. Given a set of symmetric indices, this
/// function defines the canonical order of the `TensorIndex`s assigned to them
/// to be such that the lowest index positions take any generic spatial tensor
/// indices (e.g. `ti::i`, `ti::j`), the next lowest index positions take any
/// generic spacetime indices (e.g. `ti::a`, `ti::b`), and the highest index
/// positions take any concrete time indices (e.g. `ti::t`, `ti::T`). We can
/// imagine the canonical ordering of symmetric indices to look generally like:
/// `[spatial indices ... | spacetime indices... | time indices...]`.
///
/// Within the subsets of spatial, spacetime, and time indices, the
/// `TensorIndex`s in each will be ordered such that lowercase indices come
/// before uppercase, where both are ordered alphabetically. Another way of
/// saying this is that if we had a rank N `Tensor` that was fully symmetric,
/// its canonical ordering would take the following form:
///
/// ```
/// [ti::i, ti::j, ti::k, ..., ti::I, ti::J, ti::K, ...,  // spatial
///  ti::a, ti::b, ti::c, ..., ti::A, ti::B, ti::C, ...,  // spacetime
///  ti::t, ti::t, ti::t, ..., ti::T, ti::T, ti::T, ...]  // time
/// ```
///
/// Here are some examples:
///
/// ```
/// symmetry: <1, 1, 1>
/// set of `TensorIndex`s: {ti::t, ti::i, ti::a}
/// canonical ordering: [ti::i, ti::a, ti::t]
///
/// symmetry: <1, 1, 1>
/// set of `TensorIndex`s: {ti::A, ti::a, ti::b}
/// canonical ordering: [ti::a, ti::b, ti::A]
/// ```
///
/// When a `Tensor` is not fully symmetric, the `TensorIndex` labels for any
/// indices that do not have symmetry with any other will simply keep their
/// label because it cannot be swapped:
///
/// ```
/// symmetry: <1, 2, 1>
/// set of `TensorIndex`s: {ti::a, ti::b, ti::i}
/// canonical ordering: [ti::i, ti::b, ti::a]
///
/// symmetry: <2, 1, 1>
/// set of `TensorIndex`s: {ti::t, ti::k, ti::j}
/// canonical ordering: [ti::t, ti::j, ti::k]
/// ```
///
/// If there is more than one set of symmetric indices, each of the subsets are
/// individually reordered:
///
/// ```
/// symmetry: <1, 2, 2, 1>
/// set of `TensorIndex`s: {ti::a, ti::b, ti::t, ti::i}
/// canonical ordering: [ti::i, ti::b, ti::t, ti::a]
/// ```
///
/// The motivation for this specific canonical reordering is to quickly assess
/// which components to assign to and which ones to skip when generic spatial
/// and/or concrete time indices are used for symmetric spacetime indices in the
/// resulting left hand side tensor when using `TensorExpression`s.
///
/// Let's take the spacetime metric \f$g_{ab}\f$ as our motivating example. This
/// tensor has symmetric spacetime indices, and let's say we only want to assign
/// to \f$g_{ti}\f$. We want to loop over all 10 independent components of
/// \f$g_{ab}\f$ and skip components outside of \f$g_{ti}\f$, e.g. \f$g_{xy}\f$
/// (or \f$g_{12}\f$) and \f$g_{tt}\f$ (or \f$g_{00}\f$). To do so, when we see
/// a multi-index like `{2, 1}` in our loop, we align `{2, 1}` with `{t, i}` and
/// ask if the `2` is a valid index for `t` and if the `1` is a valid index for
/// `i`. `1` is valid for `i`, but `2` is not valid for `t`, so we correctly
/// skip over `{2, 1}` and don't assign to this component.
///
/// However, this simple logic can lead to false positives or negatives when the
/// indices are symmetric. What if the multi-index we're asking about is
/// `{0, 1}` (\f$g_{01}\f$ or \f$g_{tx}\f$)? This logic would correctly
/// determine that this is one of the components we want to assign to. But what
/// if the multi-index was `{1, 0}`? The logic would incorrectly say to skip
/// over and not assign to this multi-index because `1` is not a valid index for
/// `t` and `0` is not valid for `i`. However, because \f$g_{ab}\f$ is
/// symmetric, both `{0, 1}` and `{1, 0}` should give the same result, but
/// `{1, 0}` gives us a false negative. Moreover and more generally, assigning
/// to \f$g_{ti}\f$ and \f$g_{it}\f$ should yield the same behavior (assign to
/// the same set of components).
///
/// One way to address this would be to check the multi-indices for all
/// permutations of symmetric index values, e.g. is `{0, 1}` *or* `{1, 0}`
/// valid? And if so, then evaluate it. However, this adds work at runtime and
/// the number of permutations to check increases as we increase the number of
/// symmetric indices.
///
/// The canonical reordering done by *this* function solves this problem by
/// reordering the `TensorIndex`s to align nicely with the canonical multi-index
/// ordering implemented by
/// `::Tensor_detail::Structure::get_canonical_tensor_index`.
/// `get_canonical_tensor_index` takes a storage index (which corresponds to an
/// independent tensor component) and returns a canonical multi-index. This
/// canonical multi-index is such that index values for symmetric indices will
/// be ordered to increase from right to left. For example, for a rank 2
/// symmetric tensor, `{1, 0}` is the canonical multi-index corresponding to the
/// dependent multi-indices `{0, 1}` and `{1, 0}`. Therefore, `{1, 0}` would be
/// the multi-index returned by `get_canonical_tensor_index` that corresponds to
/// the single independent component. In other words, when we loop over the
/// independent canonical multi-indices, we are looping over the multi-index
/// permutations that are in the lower triangle of the N-dimensional matrix
/// containing all multi-index permutations. The canonical reordering of LHS
/// `TensorIndex`s for symmetric indices that is done by *this* function is
/// implemented to match this: by making time indices the rightmost, then
/// spacetime the next rightmost, and then spatial indices leftmost, we
/// guarantee that looping over the lower triangle permutations given by
/// `get_canonical_tensor_index` will not produce false positives or negatives
/// using the earlier simple logic to check for valid multi-indices.
///
/// We can use the spacetime metric as an example to demonstrate this. The
/// lower triangle multi-indices that are looped over are ordered with index
/// values increasing right to left, e.g. `{0, 0}`, `{1, 0}`, `{2, 0}`,
/// `{2, 1}`, etc. If a user wants to  assign to \f$g_{ti}\f$, then after this
/// function internally reorders the LHS indices to \f$g_{it}\f$, when we loop
/// and encounter `{1, 0}`, we correctly get that we should evaluate this
/// component without having to check its other permutation. Likewise, if a user
/// wants to assign to \f$g_{it}\f$, no reordering is done and we get the same
/// correct behavior.
///
/// This function works in general because:
/// - any `0`s in the symmetric indices will "first be dealt" to any time
///   `TensorIndex`s in the rightmost index positions and then any spacetime
///   `TensorIndex`s, where `0` is correctly valid for both, but
/// - if there are more `0`s than time + spacetime `TensorIndex`s, they will be
///   "dealt" to spatial indices, which is always correctly invalid, and
/// - if there are more time `TensorIndex`s than `0`s, values `> 0` will be
///   "dealt" to time indices, which is also always correctly invalid.
///
/// In this way, we don't ever have to check other permutations of sets of
/// symmetric index values.
///
/// \tparam LhsTensorIndices the `TensorIndex`s of the `Tensor`, e.g. `ti::a`,
/// `ti::b`, `ti::c`
/// \param canonical_symmetry the canonicalized symmetry values of the tensor
/// (see `Symmetry` for definition of the canonical ordering of symmetry values)
/// \return reordered values of `LhsTensorIndices::value...`
template <typename... LhsTensorIndices, size_t NumIndices>
constexpr std::array<size_t, NumIndices> get_reordered_tensorindex_values(
    const std::array<std::int32_t, NumIndices>& canonical_symmetry) {
  constexpr std::array<size_t, NumIndices> lhs_tensorindex_values = {
      {LhsTensorIndices::value...}};
  if constexpr (NumIndices < 2) {
    return lhs_tensorindex_values;
  } else {
    const auto compare = [](const size_t tensorindex_value1,
                            const size_t tensorindex_value2) {
      // clang-tidy thinks these two branches are the same but they aren't:
      //   if (tensorindex_value2 == ti::T.value)
      //   if (tensorindex_value2 == ti::t.value)
      // NOLINTNEXTLINE (clang-tidy: bugprone-branch-clone)
      if (tensorindex_value2 == ti::T.value) {
        return false;
      } else if (is_time_index_value(tensorindex_value1)) {
        return true;
      } else if (tensorindex_value2 == ti::t.value) {
        return false;
      }

      return (is_generic_spacetime_index_value(tensorindex_value1) and
              is_generic_spatial_index_value(tensorindex_value2)) or
             (tensorindex_value1 > tensorindex_value2 and
              is_generic_spacetime_index_value(tensorindex_value1) ==
                  is_generic_spacetime_index_value(tensorindex_value2));
    };

    std::array<size_t, NumIndices> reordered_lhs_tensorindex_values =
        lhs_tensorindex_values;

    std::int32_t max_symm_value = *alg::max_element(canonical_symmetry);
    std::int32_t symm_value_to_find = 1;
    while (symm_value_to_find <= max_symm_value) {
      size_t i = NumIndices - 1;
      while (true) {
        // skip forward until we get to the position with the value we want
        while (i > 0 and canonical_symmetry[i] != symm_value_to_find) {
          i--;
        }
        if (i == 0) {
          break;
        }

        size_t max_tensorindex_value = reordered_lhs_tensorindex_values[i];
        size_t max_index = i;

        size_t j = i - 1;
        // note: because we need to hit 0 and size_t wraps around to max size_t
        while (j < NumIndices) {
          const std::int32_t compare_symm_value = canonical_symmetry[j];
          const size_t compare_tensorindex_value =
              reordered_lhs_tensorindex_values[j];
          if (compare_symm_value == symm_value_to_find and
              compare(compare_tensorindex_value, max_tensorindex_value)) {
            max_tensorindex_value = compare_tensorindex_value;
            max_index = j;
          }
          j--;
        }
        reordered_lhs_tensorindex_values[max_index] =
            reordered_lhs_tensorindex_values[i];
        reordered_lhs_tensorindex_values[i] = max_tensorindex_value;
        i--;
      }
      symm_value_to_find++;
    }

    return reordered_lhs_tensorindex_values;
  }
}

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Evaluate subtrees of the RHS expression or the RHS expression as a
 * whole and assign the result to the LHS tensor
 *
 * \details This is for internal use only and should never be directly called.
 * See `tenex::evaluate` and use it, instead.
 *
 * `EvaluateSubtrees` controls whether we wish to evaluate RHS subtrees or the
 * entire RHS expression as one expression. See`TensorExpression` documentation
 * on equation splitting for more details on what this means.
 *
 * If `EvaluateSubtrees == false`, then it's safe if the LHS tensor is used in
 * the RHS expression, so long as the generic index orders are the same. This
 * means that the callee of this function needs to first verify this is true
 * before calling this function. Under these conditions, this is a safe
 * operation because the implementation modifies each LHS component once and
 * does not revisit and access any LHS components after they've been updated.
 * For example, say we do `tenex::evaluate<ti_a, ti_b>(make_not_null(&L),
 * 5.0 * L(ti_a, ti_b));`. This function will first compute the RHS for some
 * concrete LHS, e.g. \f$L_{00}\f$. To compute this, it accesses \f$L_{00}\f$
 * in the RHS tree, multiplies it by `5.0`, then updates \f$L_{00}\f$ to be the
 * result of this multiplication. Next, it might compute \f$L_{01}\f$, where
 * only \f$L_{01}\f$ is accessed, and which hasn't yet been modified. Then the
 * next component is computed and updated, and so forth. These steps are
 * performed once for each unique LHS index. Therefore, it is important to note
 * that this kind of operation being safe to perform is
 * implementation-dependent. Specifically, the safety of the operation depends
 * on the order of LHS component access and assignment.
 *
 * \note `LhsTensorIndices` must be passed by reference because non-type
 * template parameters cannot be class types until C++20.
 *
 * @tparam EvaluateSubtrees whether or not to evaluate subtrees of RHS
 * expression
 * @tparam LhsTensorIndices the `TensorIndex`s of the `Tensor` on the LHS of the
 * tensor expression, e.g. `ti::a`, `ti::b`, `ti::c`
 * @param lhs_tensor pointer to the resultant LHS `Tensor` to fill
 * @param rhs_tensorexpression the RHS TensorExpression to be evaluated
 */
template <bool EvaluateSubtrees, typename... LhsTensorIndices,
          typename LhsDataType, typename LhsSymmetry, typename LhsIndexList,
          typename Derived, typename RhsDataType, typename RhsSymmetry,
          typename RhsIndexList, typename... RhsTensorIndices,
          size_t... LhsInts>
void evaluate_impl(
    const gsl::not_null<Tensor<LhsDataType, LhsSymmetry, LhsIndexList>*>
        lhs_tensor,
    const TensorExpression<Derived, RhsDataType, RhsSymmetry, RhsIndexList,
                           tmpl::list<RhsTensorIndices...>>&
        rhs_tensorexpression,
    const std::index_sequence<LhsInts...>& /*lhs_ints*/) {
  constexpr size_t num_lhs_indices = sizeof...(LhsTensorIndices);
  constexpr size_t num_rhs_indices = sizeof...(RhsTensorIndices);

  using lhs_tensorindex_list = tmpl::list<LhsTensorIndices...>;
  using rhs_tensorindex_list = tmpl::list<RhsTensorIndices...>;

  using lhs_tensor_type = typename std::decay_t<decltype(*lhs_tensor)>;

  static_assert(is_supported_tensor_datatype_v<LhsDataType> and
                    is_supported_tensor_datatype_v<RhsDataType>,
                "TensorExpressions currently only support Tensors whose data "
                "type is double, std::complex<double>, DataVector, or "
                "ComplexDataVector. It is possible to add support for other "
                "data types that are supported by Tensor.");
  static_assert(
      is_assignable_v<LhsDataType, RhsDataType>,
      "Assignment of the LHS Tensor's data type to the RHS TensorExpression's "
      "data type is not supported. This happens from doing something like e.g. "
      "trying to assign a Tensor<double> to a Tensor<DataVector> or a "
      "Tensor<DataVector> to a Tensor<ComplexDataVector>.");
  // `Symmetry` currently prevents this because antisymmetries are not currently
  // supported for `Tensor`s. This check is repeated here because if
  // antisymmetries are later supported for `Tensor`, using antisymmetries in
  // `TensorExpression`s will not automatically work. The implementations of the
  // derived `TensorExpression` types assume no antisymmetries (assume positive
  // `Symmetry` values), so support for antisymmetries in `TensorExpression`s
  // will still need to be implemented.
  static_assert(CheckNoLhsAntiSymmetries<LhsSymmetry>::value,
                "Anti-symmetric Tensors are not currently supported by "
                "TensorExpressions.");
  static_assert(
      tmpl::equal_members<
          typename remove_time_indices<lhs_tensorindex_list>::type,
          typename remove_time_indices<rhs_tensorindex_list>::type>::value,
      "The generic indices on the LHS of a tensor equation (that is, the "
      "template parameters specified in evaluate<...>) must match the generic "
      "indices of the RHS TensorExpression. This error occurs as a result of a "
      "call like evaluate<ti::a, ti::b>(R(ti::A, ti::b) * S(ti::a, ti::c)), "
      "where the generic indices of the evaluated RHS expression are ti::b and "
      "ti::c, but the generic indices provided for the LHS are ti::a and "
      "ti::b.");
  static_assert(
      tensorindex_list_is_valid<lhs_tensorindex_list>::value,
      "Cannot assign a tensor expression to a LHS tensor with a repeated "
      "generic index, e.g. evaluate<ti::a, ti::a>. (Note that the concrete "
      "time indices (ti::T and ti::t) can be repeated.)");
  static_assert(
      not contains_indices_to_contract<num_lhs_indices>(
          {{LhsTensorIndices::value...}}),
      "Cannot assign a tensor expression to a LHS tensor with generic "
      "indices that would be contracted, e.g. evaluate<ti::A, ti::a>.");
  // `IndexPropertyCheck` does also check that valence (Up/Lo) of indices that
  // correspond in the RHS and LHS tensors are equal, but the assertion message
  // below does not mention this because a mismatch in valence should have been
  // caught due to the combination of (i) the Tensor::operator() assertion
  // checking that generic indices' valences match the tensor's indices'
  // valences and (ii) the above assertion that RHS and LHS generic indices
  // match
  static_assert(
      IndexPropertyCheck<LhsIndexList, RhsIndexList, lhs_tensorindex_list,
                         rhs_tensorindex_list>::value,
      "At least one index of the tensor evaluated from the RHS expression "
      "cannot be evaluated to its corresponding index in the LHS tensor. This "
      "is due to a difference in number of spatial dimensions or Frame type "
      "between the index on the RHS and LHS. "
      "e.g. evaluate<ti::a, ti::b>(L, R(ti::b, ti::a));, where R's first "
      "index has 2 spatial dimensions but L's second index has 3 spatial "
      "dimensions. Check RHS and LHS indices that use the same generic index.");
  static_assert(Derived::height_relative_to_closest_tensor_leaf_in_subtree <
                    std::numeric_limits<size_t>::max(),
                "Either no Tensors were found in the RHS TensorExpression or "
                "the depth of the tree exceeded the maximum size_t value (very "
                "unlikely). If there is indeed a Tensor in the RHS expression "
                "and assuming the tree's height is not actually the maximum "
                "size_t value, then there is a flaw in the logic for computing "
                "the derived TensorExpression types' member, "
                "height_relative_to_closest_tensor_leaf_in_subtree.");

  if constexpr (EvaluateSubtrees) {
    // Make sure the LHS tensor doesn't also appear in the RHS tensor expression
    (~rhs_tensorexpression).assert_lhs_tensor_not_in_rhs_expression(lhs_tensor);
    // If the LHS data type is a vector, size the LHS tensor components if their
    // size does not match the size from a `Tensor` in the RHS expression
    if constexpr (is_derived_of_vector_impl_v<LhsDataType>) {
      const size_t rhs_component_size =
          (~rhs_tensorexpression).get_rhs_tensor_component_size();
      if (rhs_component_size != (*lhs_tensor)[0].size()) {
        for (auto& lhs_component : *lhs_tensor) {
          lhs_component = LhsDataType(rhs_component_size);
        }
      }
    }
  }

  constexpr std::array<std::int32_t, num_lhs_indices> lhs_symmetry = {
      {tmpl::at_c<LhsSymmetry, LhsInts>::value...}};
  constexpr std::array<size_t, num_lhs_indices> reordered_tensorindex_values =
      get_reordered_tensorindex_values<LhsTensorIndices...>(lhs_symmetry);
  using reordered_lhs_tensorindex_list =
      tmpl::list<TensorIndex<reordered_tensorindex_values[LhsInts]>...>;

  constexpr std::array<size_t, num_rhs_indices> index_transformation =
      compute_tensorindex_transformation<num_lhs_indices, num_rhs_indices>(
          reordered_tensorindex_values, {{RhsTensorIndices::value...}});

  // positions of indices in LHS tensor where generic spatial indices are used
  // for spacetime indices
  constexpr auto lhs_spatial_spacetime_index_positions =
      get_spatial_spacetime_index_positions<LhsIndexList,
                                            reordered_lhs_tensorindex_list>();
  // positions of indices in RHS tensor where generic spatial indices are used
  // for spacetime indices
  constexpr auto rhs_spatial_spacetime_index_positions =
      get_spatial_spacetime_index_positions<RhsIndexList,
                                            rhs_tensorindex_list>();

  // positions of indices in LHS tensor where concrete time indices are used
  constexpr auto lhs_time_index_positions =
      get_time_index_positions<reordered_lhs_tensorindex_list>();

  using rhs_expression_type =
      typename std::decay_t<decltype(~rhs_tensorexpression)>;

  for (size_t i = 0; i < lhs_tensor_type::size(); i++) {
    auto lhs_multi_index =
        lhs_tensor_type::structure::get_canonical_tensor_index(i);
    if (is_evaluated_lhs_multi_index(lhs_multi_index,
                                     lhs_spatial_spacetime_index_positions,
                                     lhs_time_index_positions)) {
      for (size_t j = 0; j < lhs_spatial_spacetime_index_positions.size();
           j++) {
        gsl::at(lhs_multi_index,
                gsl::at(lhs_spatial_spacetime_index_positions, j)) -= 1;
      }
      auto rhs_multi_index =
          transform_multi_index(lhs_multi_index, index_transformation);
      for (size_t j = 0; j < rhs_spatial_spacetime_index_positions.size();
           j++) {
        gsl::at(rhs_multi_index,
                gsl::at(rhs_spatial_spacetime_index_positions, j)) += 1;
      }

      // The expression will either be evaluated as one whole expression
      // or it will be split up into subtrees that are evaluated one at a time.
      // See the section on splitting in the documentation for the
      // `TensorExpression` class to understand the logic and terminology used
      // in this control flow below.
      if constexpr (EvaluateSubtrees) {
        // the expression is split up, so evaluate subtrees at splits
        (~rhs_tensorexpression)
            .evaluate_primary_subtree((*lhs_tensor)[i], rhs_multi_index);
        if constexpr (not rhs_expression_type::is_primary_start) {
          // the root expression type is not the starting point of a leg, so it
          // has not yet been evaluated, so now we evaluate this last leg of the
          // expression at the root of the tree
          (*lhs_tensor)[i] =
              (~rhs_tensorexpression)
                  .get_primary((*lhs_tensor)[i], rhs_multi_index);
        }
      } else {
        // the expression is not split up, so evaluate full expression
        (*lhs_tensor)[i] = (~rhs_tensorexpression).get(rhs_multi_index);
      }
    }
  }
}

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Assign a value to components of the LHS tensor
 *
 * \details This is for internal use only and should never be directly called.
 * See `tenex::evaluate` and use it, instead.
 *
 * \note `LhsTensorIndices` must be passed by reference because non-type
 * template parameters cannot be class types until C++20.
 *
 * @tparam LhsTensorIndices the `TensorIndex`s of the `Tensor` on the LHS of the
 * tensor expression, e.g. `ti::a`, `ti::b`, `ti::c`
 * @param lhs_tensor pointer to the resultant LHS `Tensor` to fill
 * @param rhs_value the RHS value to assigned
 */
template <typename... LhsTensorIndices, typename X, typename LhsSymmetry,
          typename LhsIndexList, typename NumberType, size_t... LhsInts>
void evaluate_impl(
    const gsl::not_null<Tensor<X, LhsSymmetry, LhsIndexList>*> lhs_tensor,
    const NumberType& rhs_value,
    const std::index_sequence<LhsInts...>& /*lhs_ints*/) {
  using lhs_tensor_type = typename std::decay_t<decltype(*lhs_tensor)>;
  constexpr size_t num_lhs_indices = sizeof...(LhsTensorIndices);
  using lhs_tensorindex_list = tmpl::list<LhsTensorIndices...>;

  static_assert(is_supported_tensor_datatype_v<X> and
                "TensorExpressions currently only support Tensors whose data "
                "type is double, std::complex<double>, DataVector, or "
                "ComplexDataVector. It is possible to add support for other "
                "data types that are supported by Tensor.");
  static_assert(
      is_assignable_v<X, NumberType>,
      "Assignment of the LHS Tensor's data type to the RHS number's data type "
      "is not supported within TensorExpressions. This happens from doing "
      "something like e.g. trying to assign a double to a DataVector or a "
      "DataVector to a ComplexDataVector.");
  // `Symmetry` currently prevents this because antisymmetries are not currently
  // supported for `Tensor`s. This check is repeated here because if
  // antisymmetries are later supported for `Tensor`, using antisymmetries in
  // `TensorExpression`s will not automatically work. The implementations of the
  // derived `TensorExpression` types assume no antisymmetries (assume positive
  // `Symmetry` values), so support for antisymmetries in `TensorExpression`s
  // will still need to be implemented.
  static_assert(CheckNoLhsAntiSymmetries<LhsSymmetry>::value,
                "Anti-symmetric Tensors are not currently supported by "
                "TensorExpressions.");
  static_assert(
      tensorindex_list_is_valid<lhs_tensorindex_list>::value,
      "Cannot assign a tensor expression to a LHS tensor with a repeated "
      "generic index, e.g. evaluate<ti::a, ti::a>. (Note that the concrete "
      "time indices (ti::T and ti::t) can be repeated.)");
  static_assert(
      not contains_indices_to_contract<num_lhs_indices>(
          {{LhsTensorIndices::value...}}),
      "Cannot assign a tensor expression to a LHS tensor with generic "
      "indices that would be contracted, e.g. evaluate<ti::A, ti::a>.");

  if constexpr (is_derived_of_vector_impl_v<X>) {
    ASSERT(get_size((*lhs_tensor)[0]) > 0,
           "Tensors with vector components must be sized before calling "
           "\ntenex::evaluate<...>("
           "\n\tgsl::not_null<Tensor<VectorType, ...>*>, number).");
  }

  constexpr std::array<std::int32_t, num_lhs_indices> lhs_symmetry = {
      {tmpl::at_c<LhsSymmetry, LhsInts>::value...}};
  constexpr std::array<size_t, num_lhs_indices> reordered_tensorindex_values =
      get_reordered_tensorindex_values<LhsTensorIndices...>(lhs_symmetry);
  (void)reordered_tensorindex_values;  // silence false unused variable warning
  using reordered_lhs_tensorindex_list =
      tmpl::list<TensorIndex<reordered_tensorindex_values[LhsInts]>...>;

  // positions of indices in LHS tensor where generic spatial indices are used
  // for spacetime indices
  constexpr auto lhs_spatial_spacetime_index_positions =
      get_spatial_spacetime_index_positions<LhsIndexList,
                                            reordered_lhs_tensorindex_list>();

  // positions of indices in LHS tensor where concrete time indices are used
  constexpr auto lhs_time_index_positions =
      get_time_index_positions<reordered_lhs_tensorindex_list>();

  for (size_t i = 0; i < lhs_tensor_type::size(); i++) {
    auto lhs_multi_index =
        lhs_tensor_type::structure::get_canonical_tensor_index(i);
    if (is_evaluated_lhs_multi_index(lhs_multi_index,
                                     lhs_spatial_spacetime_index_positions,
                                     lhs_time_index_positions)) {
      (*lhs_tensor)[i] = rhs_value;
    }
  }
}
}  // namespace detail

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Assign the result of a RHS tensor expression to a tensor with the LHS
 * index order set in the template parameters
 *
 * \details Uses the right hand side (RHS) TensorExpression's index ordering
 * (`RhsTE::args_list`) and the desired left hand side (LHS) tensor's index
 * ordering (`LhsTensorIndices`) to fill the provided LHS Tensor with that LHS
 * index ordering. This can carry out the evaluation of a RHS tensor expression
 * to a LHS tensor with the same index ordering, such as \f$L_{ab} = R_{ab}\f$,
 * or different ordering, such as \f$L_{ba} = R_{ab}\f$.
 *
 * The symmetry of the provided LHS Tensor need not match the symmetry
 * determined from evaluating the RHS TensorExpression according to its order of
 * operations. This allows one to specify LHS symmetries (via `lhs_tensor`) that
 * may not be preserved by the RHS expression's order of operations, which
 * depends on how the expression is written and implemented.
 *
 * The LHS `Tensor` cannot be part of the RHS expression, e.g.
 * `evaluate(make_not_null(&L), L() + R());`, because the LHS `Tensor` will
 * generally not be computed correctly when the RHS `TensorExpression` is split
 * up and the LHS tensor components are computed by accumulating the result of
 * subtrees (see the section on splitting in the documentation for the
 * `TensorExpression` class). If you need to use the LHS `Tensor` on the RHS,
 * use `tenex::update` instead.
 *
 * ### Example usage
 * Given `Tensor`s `R`, `S`, `T`, `G`, and `H`, we can compute the LHS tensor
 * \f$L\f$ in the equation \f$L_{a} = R_{ab} S^{b} + G_{a} - H_{ba}{}^{b} T\f$
 * by doing:
 *
 * \snippet Test_MixedOperations.cpp use_evaluate_with_result_as_arg
 *
 * \note `LhsTensorIndices` must be passed by reference because non-type
 * template parameters cannot be class types until C++20.
 *
 * @tparam LhsTensorIndices the `TensorIndex`s of the `Tensor` on the LHS of the
 * tensor expression, e.g. `ti::a`, `ti::b`, `ti::c`
 * @param lhs_tensor pointer to the resultant LHS `Tensor` to fill
 * @param rhs_tensorexpression the RHS TensorExpression to be evaluated
 */
template <auto&... LhsTensorIndices, typename LhsDataType, typename LhsSymmetry,
          typename LhsIndexList, typename Derived, typename RhsDataType,
          typename RhsSymmetry, typename RhsIndexList,
          typename... RhsTensorIndices>
void evaluate(
    const gsl::not_null<Tensor<LhsDataType, LhsSymmetry, LhsIndexList>*>
        lhs_tensor,
    const TensorExpression<Derived, RhsDataType, RhsSymmetry, RhsIndexList,
                           tmpl::list<RhsTensorIndices...>>&
        rhs_tensorexpression) {
  using rhs_expression_type =
      typename std::decay_t<decltype(~rhs_tensorexpression)>;
  constexpr bool evaluate_subtrees =
      rhs_expression_type::primary_subtree_contains_primary_start;
  detail::evaluate_impl<evaluate_subtrees,
                        std::decay_t<decltype(LhsTensorIndices)>...>(
      lhs_tensor, rhs_tensorexpression,
      std::make_index_sequence<sizeof...(LhsTensorIndices)>{});
}

/// @{
/*!
 * \ingroup TensorExpressionsGroup
 * \brief Assign a number to components of a tensor with the LHS index order
 * set in the template parameters
 *
 * \details
 * Example usage:
 * \snippet Test_MixedOperations.cpp assign_double_to_index_subsets
 *
 * \note The components of the LHS `Tensor` passed in must already be sized
 * because there is no way to infer component size from the RHS
 *
 * \note `LhsTensorIndices` must be passed by reference because non-type
 * template parameters cannot be class types until C++20.
 *
 * @tparam LhsTensorIndices the `TensorIndex`s of the `Tensor` on the LHS of the
 * tensor expression, e.g. `ti::a`, `ti::b`, `ti::c`
 * @param lhs_tensor pointer to the resultant LHS `Tensor` to fill
 * @param rhs_value the RHS value to assign
 */
template <auto&... LhsTensorIndices, typename X, typename LhsSymmetry,
          typename LhsIndexList, typename N,
          Requires<std::is_arithmetic_v<N>> = nullptr>
void evaluate(
    const gsl::not_null<Tensor<X, LhsSymmetry, LhsIndexList>*> lhs_tensor,
    const N rhs_value) {
  detail::evaluate_impl<std::decay_t<decltype(LhsTensorIndices)>...>(
      lhs_tensor, rhs_value,
      std::make_index_sequence<sizeof...(LhsTensorIndices)>{});
}
template <auto&... LhsTensorIndices, typename X, typename LhsSymmetry,
          typename LhsIndexList, typename N>
void evaluate(
    const gsl::not_null<Tensor<X, LhsSymmetry, LhsIndexList>*> lhs_tensor,
    const std::complex<N>& rhs_value) {
  detail::evaluate_impl<std::decay_t<decltype(LhsTensorIndices)>...>(
      lhs_tensor, rhs_value,
      std::make_index_sequence<sizeof...(LhsTensorIndices)>{});
}
/// @}

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Assign the result of a RHS tensor expression to a tensor with the LHS
 * index order set in the template parameters
 *
 * \details Uses the right hand side (RHS) TensorExpression's index ordering
 * (`RhsTE::args_list`) and the desired left hand side (LHS) tensor's index
 * ordering (`LhsTensorIndices`) to construct a LHS Tensor with that LHS index
 * ordering. This can carry out the evaluation of a RHS tensor expression to a
 * LHS tensor with the same index ordering, such as \f$L_{ab} = R_{ab}\f$, or
 * different ordering, such as \f$L_{ba} = R_{ab}\f$.
 *
 * The symmetry of the returned LHS Tensor depends on the order of operations in
 * the RHS TensorExpression, i.e. how the expression is written. If you would
 * like to specify the symmetry of the LHS Tensor instead of it being determined
 * by the order of operations in the RHS expression, please use the other
 * `tenex::evaluate` overload that takes an empty LHS Tensor as its first
 * argument.
 *
 * ### Example usage
 * Given `Tensor`s `R`, `S`, `T`, `G`, and `H`, we can compute the LHS tensor
 * \f$L\f$ in the equation \f$L_{a} = R_{ab} S^{b} + G_{a} - H_{ba}{}^{b} T\f$
 * by doing:
 *
 * \snippet Test_MixedOperations.cpp use_evaluate_to_return_result
 *
 * \parblock
 * \note If a generic spatial index is used for a spacetime index in the RHS
 * tensor, its corresponding index in the LHS tensor type will be a spatial
 * index with the same valence, frame, and number of spatial dimensions. If a
 * concrete time index is used for a spacetime index in the RHS tensor, the
 * index will not appear in the LHS tensor (i.e. there will NOT be a
 * corresponding LHS index where only the time index of that index has been
 * computed and its spatial indices are empty).
 * \endparblock
 *
 * \parblock
 * \note `LhsTensorIndices` must be passed by reference because non-type
 * template parameters cannot be class types until C++20.
 * \endparblock
 *
 * @tparam LhsTensorIndices the TensorIndexs of the Tensor on the LHS of the
 * tensor expression, e.g. `ti::a`, `ti::b`, `ti::c`
 * @param rhs_tensorexpression the RHS TensorExpression to be evaluated
 * @return the resultant LHS Tensor with index order specified by
 * LhsTensorIndices
 */
template <auto&... LhsTensorIndices, typename RhsTE,
          Requires<std::is_base_of_v<Expression, RhsTE>> = nullptr>
auto evaluate(const RhsTE& rhs_tensorexpression) {
  using lhs_tensorindex_list =
      tmpl::list<std::decay_t<decltype(LhsTensorIndices)>...>;
  using rhs_tensorindex_list = typename RhsTE::args_list;
  using rhs_symmetry = typename RhsTE::symmetry;
  using rhs_tensorindextype_list = typename RhsTE::index_list;

  // Stores (potentially reordered) symmetry and indices needed for constructing
  // the LHS tensor, with index order specified by LhsTensorIndices
  using lhs_tensor_symm_and_indices =
      LhsTensorSymmAndIndices<rhs_tensorindex_list, lhs_tensorindex_list,
                              rhs_symmetry, rhs_tensorindextype_list>;

  Tensor<typename RhsTE::type, typename lhs_tensor_symm_and_indices::symmetry,
         typename lhs_tensor_symm_and_indices::tensorindextype_list>
      lhs_tensor{};

  evaluate<LhsTensorIndices...>(make_not_null(&lhs_tensor),
                                rhs_tensorexpression);
  return lhs_tensor;
}

/*!
 * \ingroup TensorExpressionsGroup
 * \brief If the LHS tensor is used in the RHS expression, this should be used
 * to assign a LHS tensor to the result of a RHS tensor expression that contains
 * it
 *
 * \details See documentation for `tenex::evaluate` for basic functionality.
 *
 * `tenex::update` differs from `tenex::evaluate` in that `tenex::update` should
 * be used when some LHS `Tensor` has been partially computed, and now we would
 * like to update it with a RHS expression that contains it. In other words,
 * this should be used when we would like to emulate assignment operations like
 * `LHS +=`, `LHS -=`, `LHS *=`, etc.
 *
 * One important difference to note with `tenex::update` is that it cannot split
 * up the RHS expression and evaluate subtrees, while `tenex::evaluate` can (see
 * `TensorExpression` documentation). From benchmarking, it was found that the
 * runtime of `DataVector` expressions scales poorly as we increase the number
 * of operations. For this reason, when the data type held by the tensors in the
 * expression is `DataVector`, it's best to avoid passing RHS expressions with a
 * large number of operations (e.g. an inner product that sums over many terms).
 *
 * ### Example usage
 * In implementing a large equation with many operations, we can manually break
 * up the equation and evaluate different subexpressions at a time by making one
 * initial call to `tenex::evaluate` followed by any number of calls to
 * `tenex::update` that use the LHS tensor in the RHS expression and will
 * compute the rest of the equation:
 *
 * \snippet Test_MixedOperations.cpp use_update
 *
 * \note `LhsTensorIndices` must be passed by reference because non-type
 * template parameters cannot be class types until C++20.
 *
 * @tparam LhsTensorIndices the TensorIndexs of the Tensor on the LHS of the
 * tensor expression, e.g. `ti_a`, `ti_b`, `ti_c`
 * @param lhs_tensor pointer to the resultant LHS Tensor to fill
 * @param rhs_tensorexpression the RHS TensorExpression to be evaluated
 */
template <auto&... LhsTensorIndices, typename LhsDataType, typename RhsDataType,
          typename LhsSymmetry, typename LhsIndexList, typename Derived,
          typename RhsSymmetry, typename RhsIndexList,
          typename... RhsTensorIndices>
void update(
    const gsl::not_null<Tensor<LhsDataType, LhsSymmetry, LhsIndexList>*>
        lhs_tensor,
    const TensorExpression<Derived, RhsDataType, RhsSymmetry, RhsIndexList,
                           tmpl::list<RhsTensorIndices...>>&
        rhs_tensorexpression) {
  using lhs_tensorindex_list =
      tmpl::list<std::decay_t<decltype(LhsTensorIndices)>...>;
  // Assert that each instance of the LHS tensor in the RHS tensor expression
  // uses the same generic index order that the LHS uses
  (~rhs_tensorexpression)
      .template assert_lhs_tensorindices_same_in_rhs<lhs_tensorindex_list>(
          lhs_tensor);

  detail::evaluate_impl<false, std::decay_t<decltype(LhsTensorIndices)>...>(
      lhs_tensor, rhs_tensorexpression,
      std::make_index_sequence<sizeof...(LhsTensorIndices)>{});
}
}  // namespace tenex
