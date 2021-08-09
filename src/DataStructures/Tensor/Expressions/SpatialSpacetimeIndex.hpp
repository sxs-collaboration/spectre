// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \file
/// Defines functions and metafunctions used for helping evaluate
/// TensorExpression equations where generic spatial indices are used for
/// spacetime indices

namespace TensorExpressions {
namespace detail {
template <typename State, typename Element, typename Iteration,
          typename TensorIndexList>
struct spatial_spacetime_index_positions_impl {
  using type = typename std::conditional_t<
      Element::index_type == IndexType::Spacetime and
          not tmpl::at<TensorIndexList, Iteration>::is_spacetime,
      tmpl::push_back<State, tmpl::integral_constant<size_t, Iteration::value>>,
      State>;
};

/// \brief Given a generic index list and tensor index list, returns the list of
/// positions where the generic index is spatial and the tensor index is
/// spacetime
///
/// \tparam TensorIndexList the generic index list
/// \tparam TensorIndexTypeList the list of
/// \ref SpacetimeIndex "TensorIndexType"s
template <typename TensorIndexTypeList, typename TensorIndexList>
using spatial_spacetime_index_positions = tmpl::enumerated_fold<
    TensorIndexTypeList, tmpl::list<>,
    spatial_spacetime_index_positions_impl<
        tmpl::_state, tmpl::_element, tmpl::_3, tmpl::pin<TensorIndexList>>,
    tmpl::size_t<0>>;

/// \brief Given a generic index list and tensor index list, returns the list of
/// positions where the generic index is spatial and the tensor index is
/// spacetime
///
/// \tparam TensorIndexList the generic index list
/// \tparam TensorIndexTypeList the list of
/// \ref SpacetimeIndex "TensorIndexType"s
/// \return the list of positions where the generic index is spatial and the
/// tensor index is spacetime
template <typename TensorIndexTypeList, typename TensorIndexList>
constexpr auto get_spatial_spacetime_index_positions() noexcept {
  using spatial_spacetime_index_positions_ =
      spatial_spacetime_index_positions<TensorIndexTypeList, TensorIndexList>;
  using make_list_type = std::conditional_t<
      tmpl::size<spatial_spacetime_index_positions_>::value == 0, size_t,
      spatial_spacetime_index_positions_>;
  return make_array_from_list<make_list_type>();
}

/// @{
/// \brief Given a tensor symmetry and the positions of indices where a generic
/// spatial index is used for a spacetime index, this returns the symmetry
/// after making those indices nonsymmetric with others
///
/// \details
/// Example: If `symmetry` is `[2, 1, 1, 1]` and
/// `spatial_spacetime_index_positions` is `[1]`, then position 1 is the only
/// position where a generic spatial index is used for a spacetime index. The
/// resulting symmetry will make the index at position 1 no longer be symmetric
/// with the indices at positions 2 and 3. Therefore, the resulting symmetry
/// will be equivalent to the form of `[3, 2, 1, 1]`.
///
/// Note: the symmetry returned by this function is not necessarily in the
/// canonical form specified by ::Symmetry. In reality, for the example above,
/// this function would return `[2, 3, 1, 1]`.
///
/// \param symmetry the input tensor symmetry to transform
/// \param spatial_spacetime_index_positions the positions of the indices of the
/// tensor where a generic spatial index is used for a spacetime index
/// \return the symmetry after making the `spatial_spacetime_index_positions` of
/// `symmetry` nonsymmetric with other indices
template <
    size_t NumIndices, size_t NumSpatialSpacetimeIndices,
    Requires<(NumIndices >= 2 and NumSpatialSpacetimeIndices != 0)> = nullptr>
constexpr std::array<std::int32_t, NumIndices>
get_spatial_spacetime_index_symmetry(
    const std::array<std::int32_t, NumIndices>& symmetry,
    const std::array<size_t, NumSpatialSpacetimeIndices>&
        spatial_spacetime_index_positions) noexcept {
  std::array<std::int32_t, NumIndices> spatial_spacetime_index_symmetry{};
  const std::int32_t max_symm_value =
      static_cast<std::int32_t>(*alg::max_element(symmetry));
  for (size_t i = 0; i < NumIndices; i++) {
    gsl::at(spatial_spacetime_index_symmetry, i) = gsl::at(symmetry, i);
  }
  for (size_t i = 0; i < NumSpatialSpacetimeIndices; i++) {
    gsl::at(spatial_spacetime_index_symmetry,
            gsl::at(spatial_spacetime_index_positions, i)) += max_symm_value;
  }

  return spatial_spacetime_index_symmetry;
}

template <
    size_t NumIndices, size_t NumSpatialSpacetimeIndices,
    Requires<(NumIndices < 2 or NumSpatialSpacetimeIndices == 0)> = nullptr>
constexpr std::array<std::int32_t, NumIndices>
get_spatial_spacetime_index_symmetry(
    const std::array<std::int32_t, NumIndices>& symmetry,
    const std::array<size_t, NumSpatialSpacetimeIndices>&
    /*spatial_spacetime_index_positions*/) noexcept {
  return symmetry;
}
/// @}

template <typename S, typename E>
struct replace_spatial_spacetime_indices_helper {
  using type = tmpl::replace_at<S, E, change_index_type<tmpl::at<S, E>>>;
};

// The list of indices resulting from taking `TensorIndexTypeList` and
// replacing the spacetime indices at positions `SpatialSpacetimeIndexPositions`
// with spatial indices
template <typename TensorIndexTypeList, typename SpatialSpacetimeIndexPositions>
using replace_spatial_spacetime_indices = tmpl::fold<
    SpatialSpacetimeIndexPositions, TensorIndexTypeList,
    replace_spatial_spacetime_indices_helper<tmpl::_state, tmpl::_element>>;

/// \brief Given a number of tensor indices of two tensors and the positions of
/// each tensor's spacetime indices for which a generic spatial index was used,
/// compute the shift in the multi-index values from the first tensor's
/// multi-indices to the second's
///
/// \details
/// Example: If we have \f$R_{ijk} + S_{ijk}\f$, where  \f$R\f$'s first and
/// 2nd indices are spacetime and \f$S\f$' first index and third index are
/// spacetime, let \f$i = 0\f$, \f$j = 1\f$, and \f$k = 2\f$. The multi-index
/// that represents  \f$R_{012}\f$ is `{0 + 1, 1 + 1, 2} = {1, 2, 2}` and the
/// multi-index that represents \f$S_{012}\f$ is
/// `{0 + 1, 1, 2 + 1} = {1, 1, 3}`. The function returns the element-wise
/// shift that is applied to convert the first multi-index to the other, which,
/// in this case, would be: `{1, 1, 3} - {1, 2, 2} = {0, -1, 1}`.
///
/// \tparam NumIndices number of indices of the two operands
/// \param positions1 first operand's index positions where a generic spatial
/// index is used for a spacetime index
/// \param positions2 second operand's index positions where a generic spatial
/// index is used for a spacetime index
/// \return the element-wise multi-index shift from the first operand's
/// multi-indices to the second's
template <size_t NumIndices, size_t NumPositions1, size_t NumPositions2>
constexpr std::array<std::int32_t, NumIndices>
spatial_spacetime_index_transformation_from_positions(
    const std::array<size_t, NumPositions1>& positions1,
    const std::array<size_t, NumPositions2>& positions2) noexcept {
  std::array<std::int32_t, NumIndices> transformation =
      make_array<NumIndices, std::int32_t>(0);
  for (size_t i = 0; i < NumPositions1; i++) {
    gsl::at(transformation, gsl::at(positions1, i))--;
  }
  for (size_t i = 0; i < NumPositions2; i++) {
    gsl::at(transformation, gsl::at(positions2, i))++;
  }
  return transformation;
}
}  // namespace detail
}  // namespace TensorExpressions
