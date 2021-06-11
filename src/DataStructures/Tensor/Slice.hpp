// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/range/combine.hpp>
#include <boost/tuple/tuple.hpp>
#include <cstddef>
#include <optional>

#include "DataStructures/Index.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// @{
/*!
 * \ingroup DataStructuresGroup
 * \brief Slices the data within `volume_tensor` to a codimension 1 slice. The
 * slice has a constant logical coordinate in direction `sliced_dim`,
 * slicing the volume at `fixed_index` in that dimension.  For
 * example, to get the lower boundary of `sliced_dim`, pass `0` for
 * `fixed_index`; to get the upper boundary, pass
 * `extents[sliced_dim] - 1`.
 *
 * \see add_slice_to_data
 *
 * Returns Tensor class sliced to a hypersurface.
 */
template <std::size_t VolumeDim, typename VectorType, typename... Structure>
void data_on_slice(
    const gsl::not_null<Tensor<VectorType, Structure...>*> interface_tensor,
    const Tensor<VectorType, Structure...>& volume_tensor,
    const Index<VolumeDim>& element_extents, const size_t sliced_dim,
    const size_t fixed_index) noexcept {
  const size_t interface_grid_points =
      element_extents.slice_away(sliced_dim).product();
  if (interface_tensor->begin()->size() != interface_grid_points) {
    *interface_tensor = Tensor<VectorType, Structure...>(interface_grid_points);
  }

  for (SliceIterator si(element_extents, sliced_dim, fixed_index); si; ++si) {
    for (decltype(auto) interface_and_volume_tensor_components :
         boost::combine(*interface_tensor, volume_tensor)) {
      boost::get<0>(interface_and_volume_tensor_components)[si.slice_offset()] =
          boost::get<1>(
              interface_and_volume_tensor_components)[si.volume_offset()];
    }
  }
}

template <std::size_t VolumeDim, typename VectorType, typename... Structure>
Tensor<VectorType, Structure...> data_on_slice(
    const Tensor<VectorType, Structure...>& volume_tensor,
    const Index<VolumeDim>& element_extents, const size_t sliced_dim,
    const size_t fixed_index) noexcept {
  Tensor<VectorType, Structure...> interface_tensor(
      element_extents.slice_away(sliced_dim).product());
  data_on_slice(make_not_null(&interface_tensor), volume_tensor,
                element_extents, sliced_dim, fixed_index);
  return interface_tensor;
}

template <std::size_t VolumeDim, typename VectorType, typename... Structure>
void data_on_slice(
    const gsl::not_null<std::optional<Tensor<VectorType, Structure...>>*>
        interface_tensor,
    const std::optional<Tensor<VectorType, Structure...>>& volume_tensor,
    const Index<VolumeDim>& element_extents, const size_t sliced_dim,
    const size_t fixed_index) noexcept {
  if (volume_tensor.has_value()) {
    if (not(*interface_tensor)) {
      *interface_tensor = Tensor<VectorType, Structure...>{
          element_extents.slice_away(sliced_dim).product()};
    }
    data_on_slice(make_not_null(&**interface_tensor), *volume_tensor,
                  element_extents, sliced_dim, fixed_index);
  } else {
    *interface_tensor = std::nullopt;
  }
}

template <std::size_t VolumeDim, typename VectorType, typename... Structure>
std::optional<Tensor<VectorType, Structure...>> data_on_slice(
    const std::optional<Tensor<VectorType, Structure...>>& volume_tensor,
    const Index<VolumeDim>& element_extents, const size_t sliced_dim,
    const size_t fixed_index) noexcept {
  if (volume_tensor.has_value()) {
    Tensor<VectorType, Structure...> interface_tensor(
        element_extents.slice_away(sliced_dim).product());
    data_on_slice(make_not_null(&interface_tensor), *volume_tensor,
                  element_extents, sliced_dim, fixed_index);
    return interface_tensor;
  }
  return std::nullopt;
}
/// @}
