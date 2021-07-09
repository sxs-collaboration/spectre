// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function TensorExpressions::evaluate(TensorExpression)

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include "DataStructures/Tensor/Expressions/LhsTensorSymmAndIndices.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {

namespace detail {
template <size_t NumIndices>
constexpr bool contains_indices_to_contract(
    const std::array<size_t, NumIndices>& tensorindices) noexcept {
  if constexpr (NumIndices < 2) {
    return false;
  } else {
    for (size_t i = 0; i < NumIndices - 1; i++) {
      for (size_t j = i + 1; j < NumIndices; j++) {
        if (gsl::at(tensorindices, i) ==
            get_tensorindex_value_with_opposite_valence(
                gsl::at(tensorindices, j))) {
          return true;
        }
      }
    }
    return false;
  }
}

/// \brief Helper struct for checking that a RHS tensor's index can be evaluated
/// to its corresponding index in the LHS tensor
///
/// \details
/// A RHS index's corresponding LHS index uses the same generic index, such as
/// `ti_a`. For it to be possible to evaluate a RHS index to its corresponding
/// LHS index, this checks that the following is true for the index on both
/// sides:
/// - has the same valence (`UpLo`)
/// - has the same `Frame` type
/// - has the same number of spatial dimensions (allowing for expressions that
///   use generic spatial indices for spacetime indices on either side)
///
/// \tparam LhsIndexList the LHS tensor's \ref SpacetimeIndex "TensorIndexType"
/// list
/// \tparam RhsIndexList the RHS tensor's \ref SpacetimeIndex "TensorIndexType"
/// list
/// \tparam LhsTensorIndexList the LHS tensor's generic index list
/// \tparam RhsTensorIndexList the RHS tensor's generic index list
/// \tparam CurrentLhsTensorIndex the current generic index of the LHS tensor
/// that is being checked, e.g. the type of `ti_a`
template <typename LhsIndexList, typename RhsIndexList,
          typename LhsTensorIndexList, typename RhsTensorIndexList,
          typename CurrentLhsTensorIndex>
struct EvaluateIndexCheckHelper {
  using lhs_index =
      tmpl::at<LhsIndexList,
               tmpl::index_of<LhsTensorIndexList, CurrentLhsTensorIndex>>;
  using rhs_index =
      tmpl::at<RhsIndexList,
               tmpl::index_of<RhsTensorIndexList, CurrentLhsTensorIndex>>;

  using type = std::bool_constant<
      lhs_index::ul == rhs_index::ul and
      std::is_same_v<typename lhs_index::Frame, typename rhs_index::Frame> and
      ((lhs_index::index_type == rhs_index::index_type and
        lhs_index::dim == rhs_index::dim) or
       (lhs_index::index_type == IndexType::Spacetime and
        lhs_index::dim == rhs_index::dim + 1) or
       (rhs_index::index_type == IndexType::Spacetime and
        lhs_index::dim + 1 == rhs_index::dim))>;
};

/// \brief Check that a RHS tensor's indices can be evaluated to their
/// corresponding indices in the LHS tensor
///
/// \details
/// For more details, see `EvaluateIndexCheckHelper`, which performs the check
/// for each index one at a time.
///
/// \tparam LhsIndexList the LHS tensor's \ref SpacetimeIndex "TensorIndexType"
/// list
/// \tparam RhsIndexList the RHS tensor's \ref SpacetimeIndex "TensorIndexType"
/// list
/// \tparam LhsTensorIndexList the LHS tensor's generic index list
/// \tparam RhsTensorIndexList the RHS tensor's generic index list
template <typename LhsIndexList, typename RhsIndexList,
          typename LhsTensorIndexList, typename RhsTensorIndexList>
using EvaluateIndexCheck =
    tmpl::fold<LhsTensorIndexList, tmpl::bool_<true>,
               tmpl::and_<tmpl::_state,
                          EvaluateIndexCheckHelper<
                              tmpl::pin<LhsIndexList>, tmpl::pin<RhsIndexList>,
                              tmpl::pin<LhsTensorIndexList>,
                              tmpl::pin<RhsTensorIndexList>, tmpl::_element>>>;
}  // namespace detail

/// \brief Computes a transformation from the LHS tensor's multi-indices to
/// the equivalent RHS tensor's multi-indices, according to the differences in
/// the orderings of their generic indices
///
/// \details
/// The elements of the transformation are the positions of the RHS generic
/// indices in the LHS generic indices. Put another way, for some `i`,
/// `rhs_tensorindices[i] == lhs_tensorindices[index_transformation[i]]`.
///
/// Here is an example of what the algorithm does:
///
/// Tensor equation: \f$L_{cab} = R_{abc}\f$
/// `lhs_tensorindices`:
/// \code
/// {2, 0, 1} // i.e. {c, a, b}
/// \endcode
/// `rhs_tensorindices`:
/// \code
/// {0, 1, 2} // i.e. {a, b, c}
/// \endcode
/// returned `index_transformation`:
/// \code
/// {1, 2, 0} // positions of RHS indices {a, b, c} in LHS indices {c, a, b}
/// \endcode
///
/// \tparam NumIndices the number of indices in the tensors
/// \param lhs_tensorindices the TensorIndexs of the LHS tensor
/// \param rhs_tensorindices the TensorIndexs of the RHS tensor
/// \return a transformation from the LHS tensor's multi-indices to the
/// equivalent RHS tensor's multi-indices
template <size_t NumIndices>
SPECTRE_ALWAYS_INLINE constexpr std::array<size_t, NumIndices>
compute_index_transformation(
    const std::array<size_t, NumIndices>& lhs_tensorindices,
    const std::array<size_t, NumIndices>& rhs_tensorindices) noexcept {
  std::array<size_t, NumIndices> index_transformation{};
  for (size_t i = 0; i < NumIndices; i++) {
    gsl::at(index_transformation, i) = static_cast<size_t>(std::distance(
        lhs_tensorindices.begin(),
        alg::find(lhs_tensorindices, gsl::at(rhs_tensorindices, i))));
  }
  return index_transformation;
}

/// \brief Computes the RHS tensor multi-index that is equivalent to a given
/// LHS tensor multi-index, according to the differences in the orderings of
/// their generic indices
///
/// \details
/// Here is an example of what the algorithm does:
///
/// Tensor equation: \f$L_{cab} = R_{abc}\f$
/// `index_transformation`:
/// \code
/// {1, 2, 0} // positions of RHS indices {a, b, c} in LHS indices {c, a, b}
/// \endcode
/// `lhs_multi_index`:
/// \code
/// {3, 4, 5} // i.e. c = 3, a = 4, b = 5
/// \endcode
/// returned equivalent `rhs_multi_index`:
/// \code
/// {4, 5, 3} // i.e. a = 4, b = 5, c = 3
/// \endcode
///
/// \tparam NumIndices the number of indices in the tensors
/// \param lhs_multi_index the multi-index of the LHS tensor
/// \param index_transformation the list of the positions of the RHS indices in
/// the LHS indices
/// \return the RHS tensor multi-index that is equivalent to `lhs_tensor_index`
template <size_t NumIndices>
SPECTRE_ALWAYS_INLINE constexpr std::array<size_t, NumIndices>
compute_rhs_multi_index(
    const std::array<size_t, NumIndices>& lhs_multi_index,
    const std::array<size_t, NumIndices>& index_transformation) noexcept {
  std::array<size_t, NumIndices> rhs_multi_index{};
  for (size_t i = 0; i < NumIndices; i++) {
    gsl::at(rhs_multi_index, i) =
        gsl::at(lhs_multi_index, gsl::at(index_transformation, i));
  }
  return rhs_multi_index;
}

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Evaluate a RHS tensor expression to a tensor with the LHS index order
 * set in the template parameters
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
 * ### Example usage
 * Given two rank 2 Tensors `R` and `S` with index order (a, b), add them
 * together and fill the provided resultant LHS Tensor `L` with index order
 * (b, a):
 * \code{.cpp}
 * TensorExpressions::evaluate<ti_b, ti_a>(
 *     make_not_null(&L), R(ti_a, ti_b) + S(ti_a, ti_b));
 * \endcode
 *
 * This represents evaluating: \f$L_{ba} = R_{ab} + S_{ab}\f$
 *
 * Note: `LhsTensorIndices` must be passed by reference because non-type
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
void evaluate(
    const gsl::not_null<Tensor<X, LhsSymmetry, LhsIndexList>*> lhs_tensor,
    const TensorExpression<Derived, X, RhsSymmetry, RhsIndexList,
                           tmpl::list<RhsTensorIndices...>>&
        rhs_tensorexpression) {
  using lhs_tensorindex_list =
      tmpl::list<std::decay_t<decltype(LhsTensorIndices)>...>;
  using rhs_tensorindex_list = tmpl::list<RhsTensorIndices...>;
  static_assert(
      tmpl::equal_members<lhs_tensorindex_list, rhs_tensorindex_list>::value,
      "The generic indices on the LHS of a tensor equation (that is, the "
      "template parameters specified in evaluate<...>) must match the generic "
      "indices of the RHS TensorExpression. This error occurs as a result of a "
      "call like evaluate<ti_a, ti_b>(R(ti_A, ti_b) * S(ti_a, ti_c)), where "
      "the generic indices of the evaluated RHS expression are ti_b and ti_c, "
      "but the generic indices provided for the LHS are ti_a and ti_b.");
  static_assert(
      tmpl::is_set<std::decay_t<decltype(LhsTensorIndices)>...>::value,
      "Cannot evaluate a tensor expression to a LHS tensor with a repeated "
      "generic index, e.g. evaluate<ti_a, ti_a>.");
  static_assert(
      not detail::contains_indices_to_contract<sizeof...(LhsTensorIndices)>(
          {{std::decay_t<decltype(LhsTensorIndices)>::value...}}),
      "Cannot evaluate a tensor expression to a LHS tensor with generic "
      "indices that would be contracted, e.g. evaluate<ti_A, ti_a>.");
  // `EvaluateIndexCheck` does also check that valence (Up/Lo) of indices that
  // correspond in the RHS and LHS tensors are equal, but the assertion message
  // below does not mention this because a mismatch in valence should have been
  // caught due to the combination of (i) the Tensor::operator() assertion
  // checking that generic indices' valences match the tensor's indices'
  // valences and (ii) the above assertion that RHS and LHS generic indices
  // match
  static_assert(
      detail::EvaluateIndexCheck<LhsIndexList, RhsIndexList,
                                 lhs_tensorindex_list,
                                 rhs_tensorindex_list>::value,
      "At least one index of the tensor evaluated from the RHS expression "
      "cannot be evaluated to its corresponding index in the LHS tensor. This "
      "is due to a difference in number of spatial dimensions or Frame type "
      "between the index on the RHS and LHS. "
      "e.g. evaluate<ti_a, ti_b>(L, R(ti_b, ti_a));, where R's first "
      "index has 2 spatial dimensions but L's second index has 3 spatial "
      "dimensions. Check RHS and LHS indices that use the same generic index.");

  using lhs_tensor_type = typename std::decay_t<decltype(*lhs_tensor)>;

  // positions of indices in LHS tensor where generic spatial indices are used
  // for spacetime indices
  constexpr auto lhs_spatial_spacetime_index_positions =
      detail::get_spatial_spacetime_index_positions<LhsIndexList,
                                                    lhs_tensorindex_list>();
  // positions of indices in RHS tensor where generic spatial indices are used
  // for spacetime indices
  constexpr auto rhs_spatial_spacetime_index_positions =
      detail::get_spatial_spacetime_index_positions<RhsIndexList,
                                                    rhs_tensorindex_list>();

  for (size_t i = 0; i < lhs_tensor_type::size(); i++) {
    if constexpr (lhs_spatial_spacetime_index_positions.size() == 0) {
      // either:
      // (i) RHS nor LHS uses a generic spatial index for a spacetime index, or
      // (ii) only RHS uses a generic spatial index for a spacetime index
      auto rhs_multi_index = compute_rhs_multi_index(
          lhs_tensor_type::structure::get_canonical_tensor_index(i),
          compute_index_transformation<sizeof...(RhsTensorIndices)>(
              {{std::decay_t<decltype(LhsTensorIndices)>::value...}},
              {{RhsTensorIndices::value...}}));
      for (size_t j = 0; j < rhs_spatial_spacetime_index_positions.size();
           j++) {
        gsl::at(rhs_multi_index,
                gsl::at(rhs_spatial_spacetime_index_positions, j)) += 1;
      }

      (*lhs_tensor)[i] =
          (~rhs_tensorexpression)
              .template get<RhsTensorIndices...>(rhs_multi_index);

    } else {
      // either:
      // (i) only LHS uses a generic spatial index for a spacetime index
      // (ii) both RHS and LHS use a generic spatial index for a spacetime index
      auto lhs_multi_index =
          lhs_tensor_type::structure::get_canonical_tensor_index(i);
      // Only evaluate the component at `lhs_multi_index` if it does not contain
      // the time index (0) for any spacetime indices for which a generic
      // spatial index is being used
      if (alg::none_of(lhs_spatial_spacetime_index_positions,
                       [lhs_multi_index](size_t j) {
                         return gsl::at(lhs_multi_index, j) == 0;
                       })) {
        for (size_t j = 0; j < lhs_spatial_spacetime_index_positions.size();
             j++) {
          gsl::at(lhs_multi_index,
                  gsl::at(lhs_spatial_spacetime_index_positions, j)) -= 1;
        }
        auto rhs_multi_index = compute_rhs_multi_index(
            lhs_multi_index,
            compute_index_transformation<sizeof...(RhsTensorIndices)>(
                {{std::decay_t<decltype(LhsTensorIndices)>::value...}},
                {{RhsTensorIndices::value...}}));
        for (size_t j = 0; j < rhs_spatial_spacetime_index_positions.size();
             j++) {
          gsl::at(rhs_multi_index,
                  gsl::at(rhs_spatial_spacetime_index_positions, j)) += 1;
        }

        (*lhs_tensor)[i] =
            (~rhs_tensorexpression)
                .template get<RhsTensorIndices...>(rhs_multi_index);
      }
    }
  }
}

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Evaluate a RHS tensor expression to a tensor with the LHS index order
 * set in the template parameters
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
 * `evaluate` overload that takes an empty LHS Tensor as its first argument.
 *
 * ### Example usage
 * Given two rank 2 Tensors `R` and `S` with index order (a, b), add them
 * together and generate the resultant LHS Tensor `L` with index order (b, a):
 * \code{.cpp}
 * auto L = TensorExpressions::evaluate<ti_b, ti_a>(
 *     R(ti_a, ti_b) + S(ti_a, ti_b));
 * \endcode
 * \metareturns Tensor
 *
 * This represents evaluating: \f$L_{ba} = R_{ab} + S_{ab}\f$
 *
 * Note: If a generic spatial index is used for a spacetime index in the RHS
 * tensor, its corresponding index in the LHS tensor type will be a spatial
 * index with the same valence, frame, and number of spatial dimensions.
 *
 * Note: `LhsTensorIndices` must be passed by reference because non-type
 * template parameters cannot be class types until C++20.
 *
 * @tparam LhsTensorIndices the TensorIndexs of the Tensor on the LHS of the
 * tensor expression, e.g. `ti_a`, `ti_b`, `ti_c`
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
}  // namespace TensorExpressions
