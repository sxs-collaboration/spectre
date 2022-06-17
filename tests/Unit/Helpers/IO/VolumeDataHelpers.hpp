// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <boost/iterator/transform_iterator.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"

/// Functions for testing volume data output.
namespace TestHelpers::io::VolumeData {
/// Helper function that multiplies a tensor component by a double, which is
/// typically the observation time, to generate different tensor values at
/// different times for testing.
template <typename T>
T multiply(const double obs_value, const T& component) {
  T result = component;
  for (auto& t : result) {
    t *= obs_value;
  }
  return result;
}

/// Helper function to check that volume data was written correctly.
/// This function checks the following:
///    0. That the provided observation_id is present in the file
///    1. That the grid_names provided are present in the file
///    2. That the provided bases and quadratures agree with the bases
///       and quadratures in the file.
///    3. That the expected_components are present in the file.
///    4. That the expected_components, after rescaling them by a constant
///       factor_to_rescale_components, agree with the components in the file.
///       Note: if components_comparison_precision is defined, then the
///       comparison is approximate, using *components_comparison_precision as
///       the tolerance.
template <typename DataType>
void check_volume_data(
    const std::string& h5_file_name, const uint32_t version_number,
    const std::string& group_name, const size_t observation_id,
    const double observation_value,
    const std::vector<DataType>& tensor_components_and_coords,
    const std::vector<std::string>& grid_names,
    const std::vector<std::vector<Spectral::Basis>>& bases,
    const std::vector<std::vector<Spectral::Quadrature>>& quadratures,
    const std::vector<std::vector<size_t>>& extents,
    const std::vector<std::string>& expected_components,
    const std::vector<std::vector<size_t>>& grid_data_orders,
    const std::optional<double>& components_comparison_precision,
    const double factor_to_rescale_components = 1.0) {
  h5::H5File<h5::AccessType::ReadOnly> file_read{h5_file_name};
  const auto& volume_file =
      file_read.get<h5::VolumeData>("/"s + group_name, version_number);

  CHECK(extents == volume_file.get_extents(observation_id));
  CHECK(volume_file.get_observation_value(observation_id) == observation_value);
  // Check that all of the grid names were written correctly by checking their
  // equality of elements
  const std::vector<std::string> read_grid_names =
      volume_file.get_grid_names(observation_id);
  [&read_grid_names, &grid_names]() {
    auto sortable_grid_names = grid_names;
    auto sortable_read_grid_names = read_grid_names;
    std::sort(sortable_grid_names.begin(), sortable_grid_names.end(),
              std::less<>{});
    std::sort(sortable_read_grid_names.begin(), sortable_read_grid_names.end(),
              std::less<>{});
    REQUIRE(sortable_read_grid_names == sortable_grid_names);
  }();
  // Find the order the grids were written in
  std::vector<size_t> grid_positions(read_grid_names.size());
  for (size_t i = 0; i < grid_positions.size(); i++) {
    auto grid_name = grid_names[i];
    auto position =
        std::find(read_grid_names.begin(), read_grid_names.end(), grid_name);
    // We know the grid name is in the read_grid_names because of the previous
    // so we know `position` is an actual pointer to an element
    grid_positions[i] =
        static_cast<size_t>(std::distance(read_grid_names.begin(), position));
  }
  auto read_bases = volume_file.get_bases(observation_id);
  alg::sort(read_bases, std::less<>{});
  auto read_quadratures = volume_file.get_quadratures(observation_id);
  alg::sort(read_quadratures, std::less<>{});
  // We need non-const bases and quadratures in order to sort them, and we
  // need them in their string form,
  const auto& stringify = [](const auto& bases_or_quadratures) {
    std::vector<std::vector<std::string>> local_target_data{};
    local_target_data.reserve(bases_or_quadratures.size() + 1);
    for (const auto& element_data : bases_or_quadratures) {
      std::vector<std::string> target_axis_data{};
      target_axis_data.reserve(element_data.size() + 1);
      for (const auto& axis_datum : element_data) {
        target_axis_data.emplace_back(MakeString{} << axis_datum);
      }
      local_target_data.push_back(target_axis_data);
    }
    return local_target_data;
  };
  auto target_bases = stringify(bases);
  alg::sort(target_bases, std::less<>{});
  auto target_quadratures = stringify(quadratures);
  alg::sort(target_quadratures, std::less<>{});
  CHECK(target_bases == read_bases);
  CHECK(target_quadratures == read_quadratures);

  const auto read_components =
      volume_file.list_tensor_components(observation_id);
  CHECK(alg::all_of(read_components,
                    [&expected_components](const std::string& id) {
                      return alg::found(expected_components, id);
                    }));
  // Helper Function to get number of points on a particular grid
  const auto accumulate_extents = [](const std::vector<size_t>& grid_extents) {
    return alg::accumulate(grid_extents, 1, std::multiplies<>{});
  };

  const auto read_extents = volume_file.get_extents(observation_id);
  std::vector<size_t> element_num_points(
      boost::make_transform_iterator(read_extents.begin(), accumulate_extents),
      boost::make_transform_iterator(read_extents.end(), accumulate_extents));
  const auto read_points_by_element = [&element_num_points]() {
    std::vector<size_t> read_points(element_num_points.size());
    read_points[0] = 0;
    for (size_t index = 1; index < element_num_points.size(); index++) {
      read_points[index] =
          read_points[index - 1] + element_num_points[index - 1];
    }
    return read_points;
  }();
  // Given a DataType, corresponding to contiguous data read out of a
  // file, find the data which was written by the grid whose extents are
  // found at position `grid_index` in the vector of extents.
  const auto get_grid_data = [&element_num_points, &read_points_by_element](
                                 const DataVector& all_data,
                                 const size_t grid_index) {
    DataType result(element_num_points[grid_index]);
    // clang-tidy: do not use pointer arithmetic
    std::copy(&all_data[read_points_by_element[grid_index]],
              &all_data[read_points_by_element[grid_index]] +  // NOLINT
                  element_num_points[grid_index],
              result.begin());
    return result;
  };
  // The tensor components can be written in any order to the file, we loop
  // over the expected components rather than the read components because they
  // are in a particular order.
  for (size_t i = 0; i < expected_components.size(); i++) {
    const auto& component = expected_components[i];
    // for each grid
    for (size_t j = 0; j < grid_names.size(); j++) {
      if (components_comparison_precision) {
        Approx custom_approx = Approx::custom()
                                   .epsilon(*components_comparison_precision)
                                   .scale(1.0);
        CHECK_ITERABLE_CUSTOM_APPROX(
            get_grid_data(
                volume_file.get_tensor_component(observation_id, component),
                grid_positions[j]),
            multiply(factor_to_rescale_components,
                     tensor_components_and_coords[grid_data_orders[j][i]]),
            custom_approx);
      } else {
        CHECK(get_grid_data(
                  volume_file.get_tensor_component(observation_id, component),
                  grid_positions[j]) ==
              multiply(factor_to_rescale_components,
                       tensor_components_and_coords[grid_data_orders[j][i]]));
      }
    }
  }
}
}  // namespace TestHelpers::io::VolumeData
