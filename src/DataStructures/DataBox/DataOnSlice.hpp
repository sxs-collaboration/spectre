// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/range/combine.hpp>
#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace db {
/*!
 * \ingroup DataBoxGroup
 * \brief Slices volume `Tensor`s from a `DataBox` into a `Variables`
 *
 * The slice has a constant logical coordinate in direction `sliced_dim`,
 * slicing the volume at `fixed_index` in that dimension.  For
 * example, to get the lower boundary of `sliced_dim`, pass `0` for
 * `fixed_index`; to get the upper boundary, pass
 * `extents[sliced_dim] - 1`. The last argument to the function is the typelist
 * holding the tags to slice.
 *
 * \snippet Test_DataBox.cpp data_on_slice
 */
template <size_t VolumeDim, typename TagsList, typename... TagsToSlice>
Variables<tmpl::list<TagsToSlice...>> data_on_slice(
    const db::DataBox<TagsList>& box, const Index<VolumeDim>& element_extents,
    const size_t sliced_dim, const size_t fixed_index,
    tmpl::list<TagsToSlice...> /*meta*/) noexcept {
  const size_t interface_grid_points =
      element_extents.slice_away(sliced_dim).product();
  Variables<tmpl::list<TagsToSlice...>> interface_vars(interface_grid_points);
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
    };
    swallow(
        (lambda(get<TagsToSlice>(interface_vars), db::get<TagsToSlice>(box)),
         cpp17::void_type{})...);
  }
  return interface_vars;
}
}  // namespace db
