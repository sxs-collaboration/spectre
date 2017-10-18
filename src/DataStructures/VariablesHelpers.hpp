// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines helper functions for use with Variables class

#pragma once

#include <boost/range/combine.hpp>
#include <cstddef>

#include "DataStructures/Index.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/Variables.hpp"

/*!
 * \ingroup DataStructures
 * \brief Slices the data within `vars` to a codimension 1 slice. The
 * slice has a constant grid coordinate in direction `sliced_dim`, and slices
 * the volume through the point `slice_point` of this coordinate.
 *
 * \return Variables class sliced to a hypersurface.
 */
template <std::size_t VolumeDim, typename TagsList>
Variables<TagsList> data_on_slice(const Variables<TagsList>& vars,
                                  const Index<VolumeDim>& element_extents,
                                  const size_t sliced_dim,
                                  const size_t slice_point) {
  const size_t interface_grid_points =
      element_extents.slice_away(sliced_dim).product();
  const size_t volume_grid_points = vars.number_of_grid_points();
  Variables<TagsList> interface_vars(interface_grid_points);
  for (SliceIterator slice_it(element_extents, sliced_dim, slice_point);
       slice_it; ++slice_it) {
    tmpl::for_each<TagsList>([&vars, &interface_vars, &slice_it](auto tag) {
      using Tag = tmpl::type_from<decltype(tag)>;
      const auto& variable_in_volume = vars.template get<Tag>();
      auto& variable_on_interface = interface_vars.template get<Tag>();
      for (auto&& interface_and_volume_variable :
           boost::combine(variable_on_interface, variable_in_volume)) {
        auto& interface_var = interface_and_volume_variable.template get<0>();
        const auto& volume_var =
            interface_and_volume_variable.template get<1>();
        interface_var[slice_it.slice_offset()] =
            volume_var[slice_it.volume_offset()];
      }
    });
  }
  return interface_vars;
}
