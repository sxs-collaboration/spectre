// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <unordered_set>

#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {
// template specialization for handling scalar (rank 0)
template <size_t Dim, typename VectorType>
void slice_tensor_for_subcell(
    const gsl::not_null<Tensor<VectorType, Symmetry<>, index_list<>>*>
        sliced_scalar,
    const Tensor<VectorType, Symmetry<>, index_list<>>& volume_scalar,
    const Index<Dim>& subcell_extents, size_t number_of_ghost_points,
    const Direction<Dim>& direction) {
  std::unordered_set directions_to_slice{direction};

  auto& scalar_dv = get(volume_scalar);
  auto sliced_data = evolution::dg::subcell::detail::slice_data_impl(
      gsl::make_span(scalar_dv.data(), scalar_dv.size()), subcell_extents,
      number_of_ghost_points, directions_to_slice, 0)[direction];

  std::copy(sliced_data.begin(), sliced_data.end(), get(*sliced_scalar).data());
}

/// @{
/*!
 * \brief Slice a single volume tensor for a given direction and slicing depth
 * (number of ghost points).
 *
 * Note that the last argument has the type `Direction<Dim>`, not a DirectionMap
 * (cf. `evolution::dg::subcell::slice_data`)
 *
 */
template <size_t Dim, typename VectorType, typename... Structure>
void slice_tensor_for_subcell(
    const gsl::not_null<Tensor<VectorType, Structure...>*> sliced_tensor,
    const Tensor<VectorType, Structure...>& volume_tensor,
    const Index<Dim>& subcell_extents, size_t number_of_ghost_points,
    const Direction<Dim>& direction) {
  std::unordered_set directions_to_slice{direction};

  for (size_t i = 0; i < volume_tensor.size(); i++) {
    auto& ti = volume_tensor[i];

    auto sliced_data = evolution::dg::subcell::detail::slice_data_impl(
        gsl::make_span(ti.data(), ti.size()), subcell_extents,
        number_of_ghost_points, directions_to_slice, 0)[direction];

    std::copy(sliced_data.begin(), sliced_data.end(),
              (*sliced_tensor)[i].data());
  }
}

template <size_t Dim, typename VectorType, typename... Structure>
Tensor<VectorType, Structure...> slice_tensor_for_subcell(
    const Tensor<VectorType, Structure...>& volume_tensor,
    const Index<Dim>& subcell_extents, size_t number_of_ghost_points,
    const Direction<Dim>& direction) {
  Tensor<VectorType, Structure...> sliced_tensor(
      subcell_extents.slice_away(direction.dimension()).product() *
      number_of_ghost_points);
  slice_tensor_for_subcell(make_not_null(&sliced_tensor), volume_tensor,
                           subcell_extents, number_of_ghost_points, direction);
  return sliced_tensor;
}
/// @}
}  // namespace evolution::dg::subcell
