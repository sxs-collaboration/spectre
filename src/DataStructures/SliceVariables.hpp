// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <ostream>

#include "DataStructures/Index.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/Variables.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// @{
/*!
 * \ingroup DataStructuresGroup
 * \brief Slices the data within `vars` to a codimension 1 slice. The
 * slice has a constant logical coordinate in direction `sliced_dim`,
 * slicing the volume at `fixed_index` in that dimension.  For
 * example, to get the lower boundary of `sliced_dim`, pass `0` for
 * `fixed_index`; to get the upper boundary, pass
 * `extents[sliced_dim] - 1`.
 *
 * \see add_slice_to_data
 *
 * Returns Variables class sliced to a hypersurface.
 */
template <std::size_t VolumeDim, typename TagsList>
void data_on_slice(const gsl::not_null<Variables<TagsList>*> interface_vars,
                   const Variables<TagsList>& vars,
                   const Index<VolumeDim>& element_extents,
                   const size_t sliced_dim, const size_t fixed_index) noexcept {
  const size_t interface_grid_points =
      element_extents.slice_away(sliced_dim).product();
  const size_t volume_grid_points = vars.number_of_grid_points();
  constexpr const size_t number_of_independent_components =
      Variables<TagsList>::number_of_independent_components;

  if (interface_vars->number_of_grid_points() != interface_grid_points) {
    *interface_vars = Variables<TagsList>(interface_grid_points);
  }
  using value_type = typename Variables<TagsList>::value_type;
  const value_type* vars_data = vars.data();
  value_type* interface_vars_data = interface_vars->data();
  for (SliceIterator si(element_extents, sliced_dim, fixed_index); si; ++si) {
    for (size_t i = 0; i < number_of_independent_components; ++i) {
      // clang-tidy: do not use pointer arithmetic
      interface_vars_data[si.slice_offset() +                      // NOLINT
                          i * interface_grid_points] =             // NOLINT
          vars_data[si.volume_offset() + i * volume_grid_points];  // NOLINT
    }
  }
}

template <std::size_t VolumeDim, typename TagsList>
Variables<TagsList> data_on_slice(const Variables<TagsList>& vars,
                                  const Index<VolumeDim>& element_extents,
                                  const size_t sliced_dim,
                                  const size_t fixed_index) noexcept {
  Variables<TagsList> interface_vars(
      element_extents.slice_away(sliced_dim).product());
  data_on_slice(make_not_null(&interface_vars), vars, element_extents,
                sliced_dim, fixed_index);
  return interface_vars;
}
// @}

/*!
 * \ingroup DataStructuresGroup
 * \brief Adds data on a codimension 1 slice to a volume quantity. The
 * slice has a constant logical coordinate in direction `sliced_dim`,
 * slicing the volume at `fixed_index` in that dimension.  For
 * example, to add to the lower boundary of `sliced_dim`, pass `0` for
 * `fixed_index`; to add to the upper boundary, pass
 * `extents[sliced_dim] - 1`.
 *
 * \see data_on_slice
 */
template <std::size_t VolumeDim, typename TagsList>
void add_slice_to_data(const gsl::not_null<Variables<TagsList>*> volume_vars,
                       const Variables<TagsList>& vars_on_slice,
                       const Index<VolumeDim>& extents, const size_t sliced_dim,
                       const size_t fixed_index) noexcept {
  constexpr const size_t number_of_independent_components =
      Variables<TagsList>::number_of_independent_components;
  const size_t volume_grid_points = extents.product();
  const size_t slice_grid_points = extents.slice_away(sliced_dim).product();
  ASSERT(volume_vars->number_of_grid_points() == volume_grid_points,
         "volume_vars has wrong number of grid points.  Expected "
             << volume_grid_points << ", got "
             << volume_vars->number_of_grid_points());
  ASSERT(vars_on_slice.number_of_grid_points() == slice_grid_points,
         "vars_on_slice has wrong number of grid points.  Expected "
             << slice_grid_points << ", got "
             << vars_on_slice.number_of_grid_points());
  using value_type = typename Variables<TagsList>::value_type;
  value_type* const volume_data = volume_vars->data();
  const value_type* const slice_data = vars_on_slice.data();
  for (SliceIterator si(extents, sliced_dim, fixed_index); si; ++si) {
    for (size_t i = 0; i < number_of_independent_components; ++i) {
      // clang-tidy: do not use pointer arithmetic
      volume_data[si.volume_offset() + i * volume_grid_points] +=  // NOLINT
          slice_data[si.slice_offset() + i * slice_grid_points];   // NOLINT
    }
  }
}
