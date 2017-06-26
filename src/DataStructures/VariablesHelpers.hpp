// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines helper functions for use with Variables class

#pragma once

#include <cstddef>

#include "DataStructures/Index.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/Variables.hpp"

template <size_t>
class Index;

/*! \ingroup DataStructures
 *  \brief Slices the data within evolved_vars to a codimension 1 slice in the
 * sliced_dim dimension. Which surface to slice to is determined by slice_point.
 *
 * \return Variables class sliced to a hypersurface.
 */
template <std::size_t VolumeDim, typename TagsLs>
Variables<TagsLs> data_on_slice(const Variables<TagsLs>& evolved_vars,
                                const Index<VolumeDim>& element_extents,
                                const size_t& sliced_dim,
                                const size_t& slice_point) {
  const size_t interface_grid_points =
      element_extents.slice_away(sliced_dim).product();
  const size_t volume_grid_points = evolved_vars.number_of_grid_points();
  const size_t number_of_independent_components =
      evolved_vars.number_of_independent_components;
  Variables<TagsLs> interface_vars(interface_grid_points);
  const double* evolved_vars_data = evolved_vars.data();
  double* interface_vars_data = interface_vars.data();
  for (SliceIterator si(element_extents, sliced_dim, slice_point); si; ++si) {
    for (size_t i = 0; i < number_of_independent_components; ++i) {
      *(interface_vars_data + si.slice_offset() + i * interface_grid_points) =
          *(evolved_vars_data + si.volume_offset() + i * volume_grid_points);
    }
  }
  return interface_vars;
}
