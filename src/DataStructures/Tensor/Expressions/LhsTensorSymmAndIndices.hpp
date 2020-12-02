// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {
/*!
 * \ingroup TensorExpressionsGroup
 * \brief Determines and stores a LHS tensor's symmetry and index list from a
 * RHS tensor expression and desired LHS index order
 *
 * \details Given the generic index order of a RHS TensorExpression and the
 * generic index order of the desired LHS Tensor, this creates a mapping between
 * the two that is used to determine the (potentially reordered) ordering of the
 * elements of the desired LHS Tensor`s ::Symmetry and typelist of
 * \ref SpacetimeIndex "TensorIndexType"s.
 *
 * @tparam RhsTensorIndexList the typelist of TensorIndex of the RHS
 * TensorExpression, e.g. `ti_a_t`, `ti_b_t`, `ti_c_t`
 * @tparam LhsTensorIndexList the typelist of TensorIndexs of the desired LHS
 * tensor, e.g. `ti_b_t`, `ti_c_t`, `ti_a_t`
 * @tparam RhsSymmetry the ::Symmetry of the RHS indices
 * @tparam RhsTensorIndexTypeList the RHS TensorExpression's typelist of
 * \ref SpacetimeIndex "TensorIndexType"s
 */
template <typename RhsTensorIndexList, typename LhsTensorIndexList,
          typename RhsSymmetry, typename RhsTensorIndexTypeList,
          size_t NumIndices = tmpl::size<RhsSymmetry>::value,
          typename IndexSequence = std::make_index_sequence<NumIndices>>
struct LhsTensorSymmAndIndices;

template <typename RhsTensorIndexList, typename... LhsTensorIndices,
          typename RhsSymmetry, typename RhsTensorIndexTypeList,
          size_t NumIndices, size_t... Ints>
struct LhsTensorSymmAndIndices<
    RhsTensorIndexList, tmpl::list<LhsTensorIndices...>, RhsSymmetry,
    RhsTensorIndexTypeList, NumIndices, std::index_sequence<Ints...>> {
  static constexpr std::array<size_t, NumIndices> lhs_tensorindex_values = {
      {LhsTensorIndices::value...}};
  static constexpr std::array<size_t, NumIndices> rhs_tensorindex_values = {
      {tmpl::at_c<RhsTensorIndexList, Ints>::value...}};
  static constexpr std::array<size_t, NumIndices> lhs_to_rhs_map = {
      {std::distance(
          rhs_tensorindex_values.begin(),
          alg::find(rhs_tensorindex_values, lhs_tensorindex_values[Ints]))...}};

  // Desired LHS Tensor's Symmetry, typelist of TensorIndexTypes, and Structure
  using symmetry =
      Symmetry<tmpl::at_c<RhsSymmetry, lhs_to_rhs_map[Ints]>::value...>;
  using tensorindextype_list =
      tmpl::list<tmpl::at_c<RhsTensorIndexTypeList, lhs_to_rhs_map[Ints]>...>;
  using structure = Tensor_detail::Structure<
      symmetry, tmpl::at_c<RhsTensorIndexTypeList, lhs_to_rhs_map[Ints]>...>;
};
}  // namespace TensorExpressions
