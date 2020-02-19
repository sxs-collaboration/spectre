// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/range/combine.hpp>
#include <boost/tuple/tuple.hpp>
#include <cstddef>

#include "DataStructures/Index.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// @{
/*!
 * \ingroup DataStructuresGroup
 * \brief Slices volume `Tensor`s into a `Variables`
 *
 * The slice has a constant logical coordinate in direction `sliced_dim`,
 * slicing the volume at `fixed_index` in that dimension.  For
 * example, to get the lower boundary of `sliced_dim`, pass `0` for
 * `fixed_index`; to get the upper boundary, pass
 * `extents[sliced_dim] - 1`.
 */
template <typename... TagsToSlice, size_t VolumeDim>
void data_on_slice(
    const gsl::not_null<Variables<tmpl::list<TagsToSlice...>>*> interface_vars,
    const Index<VolumeDim>& element_extents, const size_t sliced_dim,
    const size_t fixed_index,
    const typename TagsToSlice::type&... tensors) noexcept {
  const size_t interface_grid_points =
      element_extents.slice_away(sliced_dim).product();
  if (interface_vars->number_of_grid_points() != interface_grid_points) {
    *interface_vars =
        Variables<tmpl::list<TagsToSlice...>>(interface_grid_points);
  }
  for (SliceIterator si(element_extents, sliced_dim, fixed_index); si; ++si) {
    const auto lambda = [&si](auto& interface_tensor,
                              const auto& volume_tensor) noexcept {
      for (decltype(auto) interface_and_volume_tensor_components :
           boost::combine(interface_tensor, volume_tensor)) {
        boost::get<0>(
            interface_and_volume_tensor_components)[si.slice_offset()] =
            boost::get<1>(
                interface_and_volume_tensor_components)[si.volume_offset()];
      }
      return '0';
    };
    expand_pack(lambda(get<TagsToSlice>(*interface_vars), tensors)...);
  }
}

template <typename... TagsToSlice, size_t VolumeDim>
Variables<tmpl::list<TagsToSlice...>> data_on_slice(
    const Index<VolumeDim>& element_extents, const size_t sliced_dim,
    const size_t fixed_index,
    const typename TagsToSlice::type&... tensors) noexcept {
  Variables<tmpl::list<TagsToSlice...>> interface_vars(
      element_extents.slice_away(sliced_dim).product());
  data_on_slice<TagsToSlice...>(make_not_null(&interface_vars), element_extents,
                                sliced_dim, fixed_index, tensors...);
  return interface_vars;
}
// @}
