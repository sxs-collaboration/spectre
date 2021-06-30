// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <iterator>
#include <utility>

#include "DataStructures/Tensor/Expressions/SpatialSpacetimeIndex.hpp"
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
 * Note: If a generic spatial index is used for a spacetime index in the RHS
 * tensor, its corresponding index in the LHS tensor type will be a spatial
 * index with the same valence, frame, and number of spatial dimensions.
 *
 * @tparam RhsTensorIndexList the typelist of TensorIndex of the RHS
 * TensorExpression
 * @tparam LhsTensorIndexList the typelist of TensorIndexs of the desired LHS
 * tensor
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
  // LHS generic indices, RHS generic indices, and the mapping between them
  static constexpr std::array<size_t, NumIndices> lhs_tensorindex_values = {
      {LhsTensorIndices::value...}};
  static constexpr std::array<size_t, NumIndices> rhs_tensorindex_values = {
      {tmpl::at_c<RhsTensorIndexList, Ints>::value...}};
  static constexpr std::array<size_t, NumIndices> lhs_to_rhs_map = {
      {std::distance(
          rhs_tensorindex_values.begin(),
          alg::find(rhs_tensorindex_values, lhs_tensorindex_values[Ints]))...}};

  // Compute symmetry of RHS after spacetime indices using generic spatial
  // indices are swapped for spatial indices
  static constexpr std::array<std::int32_t, NumIndices> rhs_symmetry = {
      {tmpl::at_c<RhsSymmetry, Ints>::value...}};
  using rhs_spatial_spacetime_index_positions_ =
      detail::spatial_spacetime_index_positions<RhsTensorIndexTypeList,
                                                RhsTensorIndexList>;
  using make_list_type = std::conditional_t<
      tmpl::size<rhs_spatial_spacetime_index_positions_>::value == 0, size_t,
      rhs_spatial_spacetime_index_positions_>;
  static constexpr auto rhs_spatial_spacetime_index_positions =
      make_array_from_list<make_list_type>();
  static constexpr std::array<std::int32_t, NumIndices>
      rhs_spatial_spacetime_index_symmetry =
          detail::get_spatial_spacetime_index_symmetry(
              rhs_symmetry, rhs_spatial_spacetime_index_positions);

  // Compute index list of RHS after spacetime indices using generic spatial
  // indices are made nonsymmetric to other indices
  using rhs_spatial_spacetime_tensorindextype_list =
      detail::replace_spatial_spacetime_indices<
          RhsTensorIndexTypeList, rhs_spatial_spacetime_index_positions_>;

  // Desired LHS Tensor's Symmetry, typelist of TensorIndexTypes, and Structure
  using symmetry =
      Symmetry<rhs_spatial_spacetime_index_symmetry[lhs_to_rhs_map[Ints]]...>;
  using tensorindextype_list =
      tmpl::list<tmpl::at_c<rhs_spatial_spacetime_tensorindextype_list,
                            lhs_to_rhs_map[Ints]>...>;
  using structure =
      Tensor_detail::Structure<symmetry,
                               tmpl::at_c<tensorindextype_list, Ints>...>;
};
}  // namespace TensorExpressions
