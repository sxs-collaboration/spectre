// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/SliceData.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace evolution::dg::subcell::detail {
namespace {
template <size_t Dim>
void copy_data(gsl::not_null<std::vector<double>*> sliced_subcell_vars,
               const gsl::span<const double>& volume_subcell_vars,
               size_t component_offset, size_t component_offset_volume,
               const std::array<size_t, Dim>& lower_bounds,
               const std::array<size_t, Dim>& upper_bounds,
               const Index<Dim>& volume_extents);

template <>
void copy_data(const gsl::not_null<std::vector<double>*> sliced_subcell_vars,
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
void copy_data(const gsl::not_null<std::vector<double>*> sliced_subcell_vars,
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
void copy_data(const gsl::not_null<std::vector<double>*> sliced_subcell_vars,
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
DirectionMap<Dim, std::vector<double>> slice_data_impl(
    const gsl::span<const double>& volume_subcell_vars,
    const Index<Dim>& subcell_extents, const size_t number_of_ghost_points,
    const DirectionMap<Dim, bool>& directions_to_slice) {
  const size_t num_pts = subcell_extents.product();
  const size_t number_of_components = volume_subcell_vars.size() / num_pts;
  std::array<size_t, Dim> result_grid_points{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(result_grid_points, d) =
        number_of_ghost_points * subcell_extents.slice_away(d).product();
  }
  DirectionMap<Dim, std::vector<double>> result{};
  for (const auto& [dir, should_slice] : directions_to_slice) {
    if (should_slice) {
      result[dir] = std::vector<double>(
          gsl::at(result_grid_points, dir.dimension()) * number_of_components);
    }
  }

  for (size_t component_index = 0; component_index < number_of_components;
       ++component_index) {
    const size_t component_offset_volume = component_index * num_pts;
    for (auto& [direction, sliced_data] : result) {
      const size_t component_offset_result =
          gsl::at(result_grid_points, direction.dimension()) * component_index;
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
      copy_data(&sliced_data, volume_subcell_vars, component_offset_result,
                component_offset_volume, lower_bounds, upper_bounds,
                subcell_extents);
    }
  }
  return result;
}

template DirectionMap<1, std::vector<double>> slice_data_impl(
    const gsl::span<const double>&, const Index<1>&, const size_t,
    const DirectionMap<1, bool>&);
template DirectionMap<2, std::vector<double>> slice_data_impl(
    const gsl::span<const double>&, const Index<2>&, const size_t,
    const DirectionMap<2, bool>&);
template DirectionMap<3, std::vector<double>> slice_data_impl(
    const gsl::span<const double>&, const Index<3>&, const size_t,
    const DirectionMap<3, bool>&);
}  // namespace evolution::dg::subcell::detail
