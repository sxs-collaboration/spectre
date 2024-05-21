// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/SliceData.hpp"

#include <array>
#include <cstddef>
#include <unordered_set>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace evolution::dg::subcell::detail {
namespace {
template <size_t Dim>
void copy_data(gsl::not_null<DataVector*> sliced_subcell_vars,
               const gsl::span<const double>& volume_subcell_vars,
               size_t component_offset, size_t component_offset_volume,
               const std::array<size_t, Dim>& lower_bounds,
               const std::array<size_t, Dim>& upper_bounds,
               const Index<Dim>& volume_extents);

template <>
void copy_data(const gsl::not_null<DataVector*> sliced_subcell_vars,
               const gsl::span<const double>& volume_subcell_vars,
               const size_t component_offset,
               const size_t component_offset_volume,
               const std::array<size_t, 1>& lower_bounds,
               const std::array<size_t, 1>& upper_bounds,
               const Index<1>& /*volume_extents*/) {
  for (size_t i = lower_bounds[0], index = 0; i < upper_bounds[0];
       ++i, ++index) {
    (*sliced_subcell_vars)[component_offset + index] =
        volume_subcell_vars[component_offset_volume + i];
  }
}

template <>
void copy_data(const gsl::not_null<DataVector*> sliced_subcell_vars,
               const gsl::span<const double>& volume_subcell_vars,
               const size_t component_offset,
               const size_t component_offset_volume,
               const std::array<size_t, 2>& lower_bounds,
               const std::array<size_t, 2>& upper_bounds,
               const Index<2>& volume_extents) {
  for (size_t j = lower_bounds[1], index = 0; j < upper_bounds[1]; ++j) {
    for (size_t i = lower_bounds[0]; i < upper_bounds[0]; ++i, ++index) {
      (*sliced_subcell_vars)[component_offset + index] =
          volume_subcell_vars[component_offset_volume + i +
                              j * volume_extents[0]];
    }
  }
}

template <>
void copy_data(const gsl::not_null<DataVector*> sliced_subcell_vars,
               const gsl::span<const double>& volume_subcell_vars,
               const size_t component_offset,
               const size_t component_offset_volume,
               const std::array<size_t, 3>& lower_bounds,
               const std::array<size_t, 3>& upper_bounds,
               const Index<3>& volume_extents) {
  for (size_t k = lower_bounds[2], index = 0; k < upper_bounds[2]; ++k) {
    for (size_t j = lower_bounds[1]; j < upper_bounds[1]; ++j) {
      for (size_t i = lower_bounds[0]; i < upper_bounds[0]; ++i, ++index) {
        (*sliced_subcell_vars)[component_offset + index] =
            volume_subcell_vars[component_offset_volume + i +
                                volume_extents[0] *
                                    (j + k * volume_extents[1])];
      }
    }
  }
}
}  // namespace

template <size_t Dim>
DirectionMap<Dim, DataVector> slice_data_impl(
    const gsl::span<const double>& volume_subcell_vars,
    const Index<Dim>& subcell_extents, const size_t number_of_ghost_points,
    const std::unordered_set<Direction<Dim>>& directions_to_slice,
    const size_t additional_buffer,
    const DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>&
        fd_to_neighbor_fd_interpolants) {
  const size_t num_pts = subcell_extents.product();
  const size_t number_of_components = volume_subcell_vars.size() / num_pts;
  std::array<size_t, Dim> result_grid_points{};
  for (size_t d = 0; d < Dim; ++d) {
    const size_t num_sliced_pts =
        number_of_ghost_points * subcell_extents.slice_away(d).product();

    ASSERT(num_sliced_pts <= num_pts,
           "Number of ghost points (" << number_of_ghost_points
                                      << ") is larger than the subcell mesh "
                                         "extent to the slicing direction ("
                                      << subcell_extents.indices().at(d)
                                      << ") ");

    gsl::at(result_grid_points, d) = num_sliced_pts;
  }
  DirectionMap<Dim, DataVector> result{};
  for (const auto& direction : directions_to_slice) {
    result[direction] =
        DataVector{gsl::at(result_grid_points, direction.dimension()) *
                       number_of_components +
                   additional_buffer};
  }

  if (fd_to_neighbor_fd_interpolants.empty()) {
    for (size_t component_index = 0; component_index < number_of_components;
         ++component_index) {
      const size_t component_offset_volume = component_index * num_pts;
      for (auto& [direction, sliced_data] : result) {
        const size_t component_offset_result =
            gsl::at(result_grid_points, direction.dimension()) *
            component_index;
        std::array<size_t, Dim> lower_bounds =
            make_array<Dim>(static_cast<size_t>(0));
        std::array<size_t, Dim> upper_bounds = subcell_extents.indices();
        if (direction.side() == Side::Lower) {
          gsl::at(upper_bounds, direction.dimension()) = number_of_ghost_points;
        } else {
          gsl::at(lower_bounds, direction.dimension()) =
              gsl::at(upper_bounds, direction.dimension()) -
              number_of_ghost_points;
        }
        // No need to worry about sliced_data including the additional buffer
        // because the instantiations of copy_data above never use the
        // sliced_data.size(). All indexing is done by the lower/upper bounds
        // arguments
        copy_data(&sliced_data, volume_subcell_vars, component_offset_result,
                  component_offset_volume, lower_bounds, upper_bounds,
                  subcell_extents);
      }
    }
  } else {
    // We add directions to `interpolated` to mark that we are interpolating
    // in this particular direction. The value of the bool is `true` _if_ we
    // did FD interpolation and `false` if DG interpolation should be
    // done. This allows passing in a DirectionalIdMap of just the neighbors
    // that need interpolation but with `std::nullopt` in order to not spend
    // resources slicing when the data would be overwritten by DG
    // interpolation (cheaper and more accurate).
    DirectionMap<Dim, bool> interpolated{};
    for (const auto& [directional_element_id, interpolant] :
         fd_to_neighbor_fd_interpolants) {
      if (LIKELY(not interpolant.has_value())) {
        // Just to keep track.
        interpolated[directional_element_id.direction()] = false;
        continue;
      }
      interpolated[directional_element_id.direction()] = true;
      auto result_span =
          gsl::make_span(result.at(directional_element_id.direction()).data(),
                         result.at(directional_element_id.direction()).size() -
                             additional_buffer);
      interpolant.value().interpolate(make_not_null(&result_span),
                                      volume_subcell_vars);
    }
    // Now copy data for neighbors that are in the same block.
    for (size_t component_index = 0; component_index < number_of_components;
         ++component_index) {
      const size_t component_offset_volume = component_index * num_pts;
      for (auto& [direction, sliced_data] : result) {
        if (UNLIKELY(interpolated.contains(direction))) {
          continue;
        }
        const size_t component_offset_result =
            gsl::at(result_grid_points, direction.dimension()) *
            component_index;
        std::array<size_t, Dim> lower_bounds =
            make_array<Dim>(static_cast<size_t>(0));
        std::array<size_t, Dim> upper_bounds = subcell_extents.indices();
        if (direction.side() == Side::Lower) {
          gsl::at(upper_bounds, direction.dimension()) = number_of_ghost_points;
        } else {
          gsl::at(lower_bounds, direction.dimension()) =
              gsl::at(upper_bounds, direction.dimension()) -
              number_of_ghost_points;
        }
        // No need to worry about sliced_data including the additional buffer
        // because the instantiations of copy_data above never use the
        // sliced_data.size(). All indexing is done by the lower/upper bounds
        // arguments
        copy_data(&sliced_data, volume_subcell_vars, component_offset_result,
                  component_offset_volume, lower_bounds, upper_bounds,
                  subcell_extents);
      }
    }
  }
  return result;
}

template DirectionMap<1, DataVector> slice_data_impl(
    const gsl::span<const double>&, const Index<1>&, const size_t,
    const std::unordered_set<Direction<1>>&, size_t,
    const DirectionalIdMap<1, std::optional<intrp::Irregular<1>>>&);
template DirectionMap<2, DataVector> slice_data_impl(
    const gsl::span<const double>&, const Index<2>&, const size_t,
    const std::unordered_set<Direction<2>>&, size_t,
    const DirectionalIdMap<2, std::optional<intrp::Irregular<2>>>&);
template DirectionMap<3, DataVector> slice_data_impl(
    const gsl::span<const double>&, const Index<3>&, const size_t,
    const std::unordered_set<Direction<3>>&, size_t,
    const DirectionalIdMap<3, std::optional<intrp::Irregular<3>>>&);
}  // namespace evolution::dg::subcell::detail
