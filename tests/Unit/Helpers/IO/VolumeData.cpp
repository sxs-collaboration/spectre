// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/IO/VolumeData.hpp"

#include <algorithm>
#include <boost/iterator/transform_iterator.hpp>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"

namespace TestHelpers::io::VolumeData {
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
    const double factor_to_rescale_components,
    const std::vector<std::string>& invalid_components) {
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
  REQUIRE(volume_file.get_bases(observation_id) == bases);
  CHECK(volume_file.get_quadratures(observation_id) == quadratures);

  {
    const auto read_components =
        volume_file.list_tensor_components(observation_id);
    CAPTURE(read_components);
    CAPTURE(expected_components);
    CHECK(alg::all_of(expected_components,
                      [&read_components](const std::string& id) {
                        return alg::found(read_components, id);
                      }));
  }
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
                                 const auto& all_data,
                                 const size_t grid_index) {
    DataType result(element_num_points[grid_index]);
    // clang-tidy: do not use pointer arithmetic
    std::copy(
        &std::get<DataType>(all_data.data)[read_points_by_element[grid_index]],
        &std::get<DataType>(
            all_data.data)[read_points_by_element[grid_index]] +  // NOLINT
            element_num_points[grid_index],
        result.begin());
    return result;
  };
  // The tensor components can be written in any order to the file, we loop
  // over the expected components rather than the read components because they
  // are in a particular order.
  for (size_t i = 0; i < expected_components.size(); i++) {
    const auto& component = expected_components[i];
    const bool component_is_invalid =
        (alg::find(invalid_components, component) != invalid_components.end());
    // for each grid
    for (size_t j = 0; j < grid_names.size(); j++) {
      const auto& data = get_grid_data(
          volume_file.get_tensor_component(observation_id, component),
          grid_positions[j]);
      if (component_is_invalid) {
        for (size_t s = 0; s < data.size(); ++s) {
          CHECK_THAT(data[s], Catch::Matchers::IsNaN());
        }
      } else if (components_comparison_precision.has_value()) {
        Approx custom_approx =
            Approx::custom()
                .epsilon(components_comparison_precision.value())
                .scale(1.0);
        CHECK_ITERABLE_CUSTOM_APPROX(
            data,
            multiply(factor_to_rescale_components,
                     tensor_components_and_coords[grid_data_orders[j][i]]),
            custom_approx);
      } else {
        CHECK(data ==
              multiply(factor_to_rescale_components,
                       tensor_components_and_coords[grid_data_orders[j][i]]));
      }
    }
  }

  // Read volume data on meshes. We previously tested all the get functions so
  // we can just use those now.
  const auto volume_data =
      volume_file.get_data_by_element(std::nullopt, std::nullopt, std::nullopt);
  size_t observations_found = 0;
  for (const auto& single_time_data : volume_data) {
    if (std::get<0>(single_time_data) != observation_id) {
      continue;
    }
    ++observations_found;
    CHECK(std::get<1>(single_time_data) == approx(observation_value));

    for (size_t j = 0; j < grid_names.size(); j++) {
      const std::string& grid_name = grid_names[j];
      // Find the element in the vector
      const auto volume_data_it = alg::find_if(
          std::get<2>(single_time_data),
          [&grid_name](const ElementVolumeData& local_volume_data) {
            return local_volume_data.element_name == grid_name;
          });
      REQUIRE(volume_data_it != std::get<2>(single_time_data).end());
      CHECK(volume_data_it->element_name == grid_name);

      for (size_t i = 0; i < expected_components.size(); i++) {
        const auto& component = expected_components[i];
        const bool component_is_valid =
            (alg::find(invalid_components, component) ==
             invalid_components.end());
        const auto expected_data = get_grid_data(
            volume_file.get_tensor_component(observation_id, component),
            grid_positions[j]);
        const auto component_data_it =
            alg::find_if(volume_data_it->tensor_components,
                         [&component](const TensorComponent& tensor_component) {
                           return tensor_component.name == component;
                         });
        REQUIRE(component_data_it != volume_data_it->tensor_components.end());
        if (component_is_valid) {
          CHECK(std::get<DataType>(component_data_it->data) == expected_data);
        }
      }
    }
  }
  CHECK(observations_found == 1);
  if (invalid_components.empty()) {
    // Test that the data is sorted by copying it, sorting it, and verifying
    // everything is the same. First sort outer vector by observation value,
    // then sort the vector of ElementVolumeData on each time slice.
    auto volume_data_sorted = volume_data;
    alg::sort(volume_data_sorted, [](const auto& lhs, const auto& rhs) {
      return std::get<1>(lhs) < std::get<1>(rhs);
    });
    for (auto& data_at_time : volume_data_sorted) {
      alg::sort(std::get<2>(data_at_time),
                [](const auto& lhs, const auto& rhs) {
                  return lhs.element_name < rhs.element_name;
                });
      for (auto& element_data : std::get<2>(data_at_time)) {
        alg::sort(element_data.tensor_components,
                  [](const auto& lhs, const auto& rhs) {
                    return lhs.name < rhs.name;
                  });
      }
    }
    CHECK((volume_data == volume_data_sorted));
  }
}

#define GET_DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                           \
  template void check_volume_data(                                       \
      const std::string& h5_file_name, const uint32_t version_number,    \
      const std::string& group_name, const size_t observation_id,        \
      const double observation_value,                                    \
      const std::vector<GET_DTYPE(data)>& tensor_components_and_coords,  \
      const std::vector<std::string>& grid_names,                        \
      const std::vector<std::vector<Spectral::Basis>>& bases,            \
      const std::vector<std::vector<Spectral::Quadrature>>& quadratures, \
      const std::vector<std::vector<size_t>>& extents,                   \
      const std::vector<std::string>& expected_components,               \
      const std::vector<std::vector<size_t>>& grid_data_orders,          \
      const std::optional<double>& components_comparison_precision,      \
      const double factor_to_rescale_components,                         \
      const std::vector<std::string>& invalid_components);

GENERATE_INSTANTIATIONS(INSTANTIATION, (DataVector, std::vector<float>))

#undef INSTANTIATION
#undef GET_DTYPE
}  // namespace TestHelpers::io::VolumeData
