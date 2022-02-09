// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>

#include "DataStructures/Index.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {
/// @{
/*!
 * \brief Slice a Variables object on subcell mesh for a given direction and
 * slicing depth (number of ghost points).
 *
 * This is essentially a wrapper of
 * `evolution::dg::subcell::detail::slice_data_impl` function, but note that the
 * last argument has the type `Direction<Dim>`, not a DirectionMap.
 */
template <size_t Dim, typename TagList>
void slice_variable_for_subcell(
    const gsl::not_null<Variables<TagList>*>& sliced_subcell_vars,
    const Variables<TagList>& volume_subcell_vars,
    const Index<Dim>& subcell_extents, const size_t ghost_zone_size,
    const Direction<Dim>& direction) {
  // check the size of sliced_subcell_vars (output)
  const size_t num_sliced_pts =
      subcell_extents.slice_away(direction.dimension()).product() *
      ghost_zone_size;
  if (sliced_subcell_vars->size() != num_sliced_pts) {
    *sliced_subcell_vars = Variables<TagList>(num_sliced_pts);
  }

  // Slice volume variables. Note that the return type of slicing function is
  // std::vector<double>.
  DirectionMap<Dim, bool> directions_to_slice{};
  directions_to_slice[direction] = true;

  std::vector<double> sliced_data{detail::slice_data_impl(
      gsl::make_span(volume_subcell_vars.data(), volume_subcell_vars.size()),
      subcell_extents, ghost_zone_size, directions_to_slice)[direction]};

  // copy the returned std::vector<double> data into sliced variables
  std::copy(sliced_data.begin(), sliced_data.end(),
            sliced_subcell_vars->data());
}

template <size_t Dim, typename TagList>
Variables<TagList> slice_variable(const Variables<TagList>& volume_subcell_vars,
                                  const Index<Dim>& subcell_extents,
                                  const size_t ghost_zone_size,
                                  const Direction<Dim>& direction) {
  Variables<TagList> sliced_subcell_vars{};
  sliced_subcell_vars.initialize(
      subcell_extents.slice_away(direction.dimension()).product() *
      ghost_zone_size);
  slice_variable_for_subcell(make_not_null(&sliced_subcell_vars),
                             volume_subcell_vars, subcell_extents,
                             ghost_zone_size, direction);
  return sliced_subcell_vars;
}
/// @}
}  // namespace evolution::dg::subcell
