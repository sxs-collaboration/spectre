// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions for evaluating `TensorExpression`s

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include "DataStructures/Tensor/Expressions/IndexPropertyCheck.hpp"
#include "DataStructures/Tensor/Expressions/LhsTensorSymmAndIndices.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndexTransformation.hpp"
#include "DataStructures/Tensor/Expressions/TimeIndex.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
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
template <bool EvaluateSubtrees, auto&... LhsTensorIndices, typename X,
          typename LhsSymmetry, typename LhsIndexList, typename Derived,
          typename RhsSymmetry, typename RhsIndexList,
          typename... RhsTensorIndices>
void evaluate_impl(
    const gsl::not_null<Tensor<X, LhsSymmetry, LhsIndexList>*> lhs_tensor,
    const TensorExpression<Derived, X, RhsSymmetry, RhsIndexList,
                           tmpl::list<RhsTensorIndices...>>&
        rhs_tensorexpression) {
  constexpr size_t num_lhs_indices = sizeof...(LhsTensorIndices);
  constexpr size_t num_rhs_indices = sizeof...(RhsTensorIndices);

  using lhs_tensorindex_list =
      tmpl::list<std::decay_t<decltype(LhsTensorIndices)>...>;
  using rhs_tensorindex_list = tmpl::list<RhsTensorIndices...>;

  static_assert(std::is_same_v<double, X> or std::is_same_v<DataVector, X>,
                "TensorExpressions currently only support Tensors whose data "
                "type is double or DataVector. It is possible to add support "
                "for other data types that are supported by Tensor.");
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
          {{std::decay_t<decltype(LhsTensorIndices)>::value...}}),
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
    // If the data type is `DataVector`, size the LHS tensor components if their
    // size does not match the size from a `Tensor` in the RHS expression
    if constexpr (std::is_same_v<DataVector, X>) {
      const size_t rhs_component_size =
          (~rhs_tensorexpression).get_rhs_tensor_component_size();
      if (rhs_component_size != (*lhs_tensor)[0].size()) {
        for (auto& lhs_component : *lhs_tensor) {
          lhs_component = DataVector(rhs_component_size);
        }
      }
    }
  }

  constexpr std::array<size_t, num_rhs_indices> index_transformation =
      compute_tensorindex_transformation<num_lhs_indices, num_rhs_indices>(
          {{std::decay_t<decltype(LhsTensorIndices)>::value...}},
          {{RhsTensorIndices::value...}});

  // positions of indices in LHS tensor where generic spatial indices are used
  // for spacetime indices
  constexpr auto lhs_spatial_spacetime_index_positions =
      get_spatial_spacetime_index_positions<LhsIndexList,
                                            lhs_tensorindex_list>();
  // positions of indices in RHS tensor where generic spatial indices are used
  // for spacetime indices
  constexpr auto rhs_spatial_spacetime_index_positions =
      get_spatial_spacetime_index_positions<RhsIndexList,
                                            rhs_tensorindex_list>();

  // positions of indices in LHS tensor where concrete time indices are used
  constexpr auto lhs_time_index_positions =
      get_time_index_positions<lhs_tensorindex_list>();

  using lhs_tensor_type = typename std::decay_t<decltype(*lhs_tensor)>;
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
 * \brief Assign a `double` value to components of the LHS tensor
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
template <auto&... LhsTensorIndices, typename X, typename LhsSymmetry,
          typename LhsIndexList>
void evaluate_impl(
    const gsl::not_null<Tensor<X, LhsSymmetry, LhsIndexList>*> lhs_tensor,
    const double rhs_value) {
  constexpr size_t num_lhs_indices = sizeof...(LhsTensorIndices);

  using lhs_tensorindex_list =
      tmpl::list<std::decay_t<decltype(LhsTensorIndices)>...>;

  static_assert(std::is_same_v<double, X> or std::is_same_v<DataVector, X>,
                "TensorExpressions currently only support Tensors whose data "
                "type is double or DataVector. It is possible to add support "
                "for other data types that are supported by Tensor.");
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
          {{std::decay_t<decltype(LhsTensorIndices)>::value...}}),
      "Cannot assign a tensor expression to a LHS tensor with generic "
      "indices that would be contracted, e.g. evaluate<ti::A, ti::a>.");

  // positions of indices in LHS tensor where generic spatial indices are used
  // for spacetime indices
  constexpr auto lhs_spatial_spacetime_index_positions =
      get_spatial_spacetime_index_positions<LhsIndexList,
                                            lhs_tensorindex_list>();

  // positions of indices in LHS tensor where concrete time indices are used
  constexpr auto lhs_time_index_positions =
      get_time_index_positions<lhs_tensorindex_list>();

  using lhs_tensor_type = typename std::decay_t<decltype(*lhs_tensor)>;

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
template <auto&... LhsTensorIndices, typename X, typename LhsSymmetry,
          typename LhsIndexList, typename Derived, typename RhsSymmetry,
          typename RhsIndexList, typename... RhsTensorIndices>
void evaluate(
    const gsl::not_null<Tensor<X, LhsSymmetry, LhsIndexList>*> lhs_tensor,
    const TensorExpression<Derived, X, RhsSymmetry, RhsIndexList,
                           tmpl::list<RhsTensorIndices...>>&
        rhs_tensorexpression) {
  using rhs_expression_type =
      typename std::decay_t<decltype(~rhs_tensorexpression)>;
  constexpr bool evaluate_subtrees =
      rhs_expression_type::primary_subtree_contains_primary_start;
  detail::evaluate_impl<evaluate_subtrees, LhsTensorIndices...>(
      lhs_tensor, rhs_tensorexpression);
}

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Assign a `double` to components of a tensor with the LHS index order
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
          typename LhsIndexList>
void evaluate(
    const gsl::not_null<Tensor<X, LhsSymmetry, LhsIndexList>*> lhs_tensor,
    const double rhs_value) {
  if constexpr (std::is_same_v<X, DataVector>) {
    ASSERT(get_size((*lhs_tensor)[0]) > 0,
           "Tensors with DataVector components must be sized before calling "
           "tenex::evaluate<...>("
           "\tgsl::not_null<Tensor<DataVector, ...>*>, double).");
  }

  detail::evaluate_impl<LhsTensorIndices...>(lhs_tensor, rhs_value);
}

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
template <auto&... LhsTensorIndices, typename X, typename LhsSymmetry,
          typename LhsIndexList, typename Derived, typename RhsSymmetry,
          typename RhsIndexList, typename... RhsTensorIndices>
void update(
    const gsl::not_null<Tensor<X, LhsSymmetry, LhsIndexList>*> lhs_tensor,
    const TensorExpression<Derived, X, RhsSymmetry, RhsIndexList,
                           tmpl::list<RhsTensorIndices...>>&
        rhs_tensorexpression) {
  using lhs_tensorindex_list =
      tmpl::list<std::decay_t<decltype(LhsTensorIndices)>...>;
  // Assert that each instance of the LHS tensor in the RHS tensor expression
  // uses the same generic index order that the LHS uses
  (~rhs_tensorexpression)
      .template assert_lhs_tensorindices_same_in_rhs<lhs_tensorindex_list>(
          lhs_tensor);

  detail::evaluate_impl<false, LhsTensorIndices...>(lhs_tensor,
                                                    rhs_tensorexpression);
}
}  // namespace tenex
