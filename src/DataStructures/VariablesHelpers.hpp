// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines helper functions for use with Variables class

#pragma once

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
  const size_t number_of_independent_components =
      vars.number_of_independent_components;
  Variables<TagsList> interface_vars(interface_grid_points);
  const double* vars_data = vars.data();
  double* interface_vars_data = interface_vars.data();
  for (SliceIterator si(element_extents, sliced_dim, slice_point); si; ++si) {
    for (size_t i = 0; i < number_of_independent_components; ++i) {
      // clang-tidy: do not use pointer arithmetic
      interface_vars_data[si.slice_offset() +                      // NOLINT
                          i * interface_grid_points] =             // NOLINT
          vars_data[si.volume_offset() + i * volume_grid_points];  // NOLINT
    }
  }
  return interface_vars;
}
