// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/ExtendConnectivityHelpers.hpp"

#include <algorithm>
#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace {

// Given a std::vector of grid_names, computes the number of blocks that exist
// and also returns a std::vector of block numbers that is a one-to-one mapping
// to each element in grid_names. The returned tuple is of the form
// [number_of_blocks, block_number_for_each_element, sorted_element_indices].
// number_of_blocks is equal to the number of blocks in the domain.
// block_number_for_each_element is a std::vector with length equal to the total
// number of grid names. sorted_element_indices is a std::vector<std::vector>
// with length equal to the number of blocks in the domain, since each subvector
// represents a given block. These subvectors are of a length equal to the
// number of elements which belong to that corresponding block.
std::tuple<size_t, std::vector<size_t>, std::vector<std::vector<size_t>>>
compute_and_organize_block_info(const std::vector<std::string>& grid_names) {
  std::vector<size_t> block_number_for_each_element;
  std::vector<std::vector<size_t>> sorted_element_indices;
  block_number_for_each_element.reserve(grid_names.size());

  // Fills block_number_for_each_element
  for (const std::string& grid_name : grid_names) {
    size_t end_position = grid_name.find(',', 1);
    block_number_for_each_element.push_back(
        static_cast<size_t>(std::stoi(grid_name.substr(2, end_position))));
  }

  const auto max_block_number =
      *std::max_element(block_number_for_each_element.begin(),
                        block_number_for_each_element.end());
  auto number_of_blocks = max_block_number + 1;
  sorted_element_indices.reserve(number_of_blocks);

  // Properly sizes subvectors of sorted_element_indices
  for (size_t i = 0; i < number_of_blocks; ++i) {
    std::vector<size_t> sizing_vector;
    auto number_of_elements_in_block =
        static_cast<size_t>(std::count(block_number_for_each_element.begin(),
                                       block_number_for_each_element.end(), i));
    sizing_vector.reserve(number_of_elements_in_block);
    sorted_element_indices.push_back(sizing_vector);
  }

  // Organizing grid_names by block
  for (size_t i = 0; i < block_number_for_each_element.size(); ++i) {
    sorted_element_indices[block_number_for_each_element[i]].push_back(i);
  }

  return std::make_tuple(number_of_blocks,
                         std::move(block_number_for_each_element),
                         std::move(sorted_element_indices));
}

// Takes in a std::vector<std::vector<size_t>> sorted_element_indices which
// houses indices associated to the elements sorted by block (labelled by the
// position of the subvector in the parent vector) and some std::vector
// property_to_sort of some element property (e.g. extents) for all elements in
// the domain. First creates a std::vector<std::vector> identical in structure
// to sorted_element_indices. Then, sorts property_to_sort by block using the
// indices for elements in each block as stored in sorted_element_indices.
template <typename T>
std::vector<std::vector<T>> sort_by_block(
    const std::vector<std::vector<size_t>>& sorted_element_indices,
    const std::vector<T>& property_to_sort) {
  std::vector<std::vector<T>> sorted_property;
  sorted_property.reserve(sorted_element_indices.size());

  // Properly sizes subvectors
  for (const auto& sorted_block_index : sorted_element_indices) {
    std::vector<T> sizing_vector;
    sizing_vector.reserve(sorted_block_index.size());
    for (const auto& sorted_element_index : sorted_block_index) {
      sizing_vector.push_back(property_to_sort[sorted_element_index]);
    }
    sorted_property.push_back(std::move(sizing_vector));
  }

  return sorted_property;
}

// Returns a std::tuple of the form
// [expected_connectivity_length, expected_number_of_grid_points, h_ref_array],
// where each of the quantities in the tuple is computed for each block
// individually. expected_connectivity_length is the expected length of the
// connectivity for the given block. expected_number_of_grid_points is the
// number of grid points that are expected to be within the block. h_ref_array
// is an array of the h-refinement in the x, y, and z directions. This function
// computes properties at the block level, as our algorithm for constructing the
// new connectivity works within a block, making it convenient to sort these
// properties early.
template <size_t SpatialDim>
std::tuple<size_t, size_t, std::array<int, SpatialDim>>
compute_block_level_properties(
    const std::vector<std::string>& block_grid_names,
    const std::vector<std::vector<size_t>>& block_extents) {
  size_t expected_connectivity_length = 0;
  // Used for reserving the length of block_logical_coords
  size_t expected_number_of_grid_points = 0;

  for (const auto& extents : block_extents) {
    size_t element_grid_points = 1;
    size_t number_of_cells_in_element = 1;
    for (size_t j = 0; j < SpatialDim; j++) {
      element_grid_points *= extents[j];
      number_of_cells_in_element *= extents[j] - 1;
    }
    // Connectivity that already exists
    expected_connectivity_length +=
        number_of_cells_in_element * pow(2, SpatialDim);
    expected_number_of_grid_points += element_grid_points;
  }

  std::string grid_name_string = block_grid_names[0];
  std::array<int, SpatialDim> h_ref_array = {};
  size_t h_ref_previous_start_position = 0;
  size_t additional_connectivity_length = 1;
  for (size_t i = 0; i < SpatialDim; ++i) {
    const size_t h_ref_start_position =
        grid_name_string.find('L', h_ref_previous_start_position + 1);
    const size_t h_ref_end_position =
        grid_name_string.find('I', h_ref_start_position);
    const int h_ref = std::stoi(
        grid_name_string.substr(h_ref_start_position + 1,
                                h_ref_end_position - h_ref_start_position - 1));
    gsl::at(h_ref_array, i) = h_ref;
    additional_connectivity_length *= pow(2, h_ref + 1) - 1;
    h_ref_previous_start_position = h_ref_start_position;
  }

  expected_connectivity_length +=
      (additional_connectivity_length - block_extents.size()) * 8;

  return std::tuple{expected_connectivity_length,
                    expected_number_of_grid_points, h_ref_array};
}

// Returns a std::vector<std::array> where each std::array represents the
// coordinates of a grid point in the block logical frame, and the entire
// std::vector is the list of all such grid points
template <size_t SpatialDim>
std::vector<std::array<double, SpatialDim>> generate_block_logical_coordinates(
    const std::vector<std::array<double, SpatialDim>>&
        element_logical_coordinates,
    const std::string& grid_name,
    const std::array<int, SpatialDim>& h_refinement_array) {
  size_t grid_points_x_start_position = 0;
  std::vector<std::array<double, SpatialDim>> block_logical_coordinates;
  block_logical_coordinates.reserve(element_logical_coordinates.size());
  std::vector<double> number_of_elements_each_direction;
  number_of_elements_each_direction.reserve(SpatialDim);
  std::vector<double> shift_each_direction;
  shift_each_direction.reserve(SpatialDim);

  // Computes number_of_elements_each_direction, element_index, and
  // shift_each_direction to each be used in the computation of the grid point
  // coordinates in the block logical frame
  for (size_t i = 0; i < SpatialDim; ++i) {
    double number_of_elements = pow(2, gsl::at(h_refinement_array, i));
    number_of_elements_each_direction.push_back(number_of_elements);
    size_t grid_points_start_position =
        grid_name.find('I', grid_points_x_start_position + 1);
    size_t grid_points_end_position =
        grid_name.find(',', grid_points_start_position);
    if (i == SpatialDim) {
      grid_points_end_position =
          grid_name.find(')', grid_points_start_position);
    }
    int element_index = std::stoi(grid_name.substr(
        grid_points_start_position + 1,
        grid_points_end_position - grid_points_start_position - 1));
    double shift = (-1 + (2 * element_index + 1) / number_of_elements);
    shift_each_direction.push_back(shift);
    grid_points_x_start_position = grid_points_start_position;
  }

  // Computes the coordinates for each grid point in the block logical frame
  for (size_t i = 0; i < element_logical_coordinates.size(); ++i) {
    std::array<double, SpatialDim> grid_point_coordinate = {};
    for (size_t j = 0; j < grid_point_coordinate.size(); ++j) {
      gsl::at(grid_point_coordinate, j) =
          1. / number_of_elements_each_direction[j] *
              element_logical_coordinates[i][j] +
          shift_each_direction[j];
    }
    block_logical_coordinates.push_back(grid_point_coordinate);
  }

  return block_logical_coordinates;
}

// Given a std::vector<double> where the elements are ordered in ascending order
// a new std::vector<double> is generated where it is a list of the original
// values in ascending order without duplicates
// Example: [1,2,2,3] -> order() -> [1,2,3]
std::vector<double> order_sorted_elements(
    const std::vector<double>& sorted_elements) {
  std::vector<double> ordered_elements;
  ordered_elements.push_back(sorted_elements[0]);
  for (size_t i = 1; i < sorted_elements.size(); ++i) {
    if (sorted_elements[i] != ordered_elements.end()[-1]) {
      ordered_elements.push_back(sorted_elements[i]);
    }
  }
  return ordered_elements;
}

// Returns a std::vector of std::pair where each std::pair is
// composed of a number for the block a given grid point resides inside of, as
// well as the grid point itself as a std::array. Generates the connectivity by
// connecting grid points (in the block logical frame) to form either
// hexahedrons, quadrilaterals, or lines depending on the SpatialDim. The
// function iteratively generates all possible shapes with all of a grid point's
// nearest neighbors. Example: Consider a 4x4 grid of evenly spaces points.
// build_connectivity_by_hexahedron generates connectivity that forms 9 sqaures.
template <size_t SpatialDim>
std::vector<std::pair<size_t, std::array<double, SpatialDim>>>
build_connectivity_by_hexahedron(const std::vector<double>& sorted_x,
                                 const std::vector<double>& sorted_y,
                                 const std::vector<double>& sorted_z,
                                 const size_t& block_number) {
  std::vector<std::pair<size_t, std::array<double, SpatialDim>>>
      connectivity_of_keys;

  std::array<double, SpatialDim> point_one = {};
  std::array<double, SpatialDim> point_two = {};
  [[maybe_unused]] std::array<double, SpatialDim> point_three = {};
  [[maybe_unused]] std::array<double, SpatialDim> point_four = {};
  [[maybe_unused]] std::array<double, SpatialDim> point_five = {};
  [[maybe_unused]] std::array<double, SpatialDim> point_six = {};
  [[maybe_unused]] std::array<double, SpatialDim> point_seven = {};
  [[maybe_unused]] std::array<double, SpatialDim> point_eight = {};

  // Algorithm for connecting grid points. Extended by if statments to account
  // for 1D, 2D, and 3D
  for (size_t i = 0; i < sorted_x.size() - 1; ++i) {
    point_one[0] = sorted_x[i];
    point_two[0] = sorted_x[i + 1];
    // 2D or 3D
    if constexpr (SpatialDim > 1) {
      point_three[0] = sorted_x[i + 1];
      point_four[0] = sorted_x[i];
      for (size_t j = 0; j < sorted_y.size() - 1; ++j) {
        point_one[1] = sorted_y[j];
        point_two[1] = sorted_y[j];
        point_three[1] = sorted_y[j + 1];
        point_four[1] = sorted_y[j + 1];
        // 3D
        if constexpr (SpatialDim == 3) {
          point_five[0] = sorted_x[i];
          point_six[0] = sorted_x[i + 1];
          point_seven[0] = sorted_x[i + 1];
          point_eight[0] = sorted_x[i];
          point_five[1] = sorted_y[j];
          point_six[1] = sorted_y[j];
          point_seven[1] = sorted_y[j + 1];
          point_eight[1] = sorted_y[j + 1];
          for (size_t k = 0; k < sorted_z.size() - 1; ++k) {
            point_one[2] = sorted_z[k];
            point_two[2] = sorted_z[k];
            point_three[2] = sorted_z[k];
            point_four[2] = sorted_z[k];
            point_five[2] = sorted_z[k + 1];
            point_six[2] = sorted_z[k + 1];
            point_seven[2] = sorted_z[k + 1];
            point_eight[2] = sorted_z[k + 1];

            connectivity_of_keys.insert(
                connectivity_of_keys.end(),
                {std::make_pair(block_number, point_one),
                 std::make_pair(block_number, point_two),
                 std::make_pair(block_number, point_three),
                 std::make_pair(block_number, point_four),
                 std::make_pair(block_number, point_five),
                 std::make_pair(block_number, point_six),
                 std::make_pair(block_number, point_seven),
                 std::make_pair(block_number, point_eight)});
          }
        } else {
          connectivity_of_keys.insert(
              connectivity_of_keys.end(),
              {std::make_pair(block_number, point_one),
               std::make_pair(block_number, point_two),
               std::make_pair(block_number, point_three),
               std::make_pair(block_number, point_four)});
        }
      }
    } else {
      connectivity_of_keys.insert(connectivity_of_keys.end(),
                                  {std::make_pair(block_number, point_one),
                                   std::make_pair(block_number, point_two)});
    }
  }
  return connectivity_of_keys;
}

// Returns the output of build_connectivity_by_hexahedron after feeding in
// specially prepared inputs. The output is the new connectivity
template <size_t SpatialDim>
std::vector<std::pair<size_t, std::array<double, SpatialDim>>>
generate_new_connectivity(
    std::vector<std::array<double, SpatialDim>>& block_logical_coordinates,
    const size_t& block_number) {
  std::vector<std::vector<double>> unsorted_coordinates;
  unsorted_coordinates.reserve(SpatialDim);

  // Takes the block_logical_coordinates and splits them up into a unique
  // std::vector for x, y, and z. These three std::vector are then stored inside
  // of a std::vector unsorted_coordinates
  for (size_t i = 0; i < SpatialDim; ++i) {
    std::vector<double> coordinates_by_direction;
    coordinates_by_direction.reserve(block_logical_coordinates.size());
    for (size_t j = 0; j < block_logical_coordinates.size(); ++j) {
      coordinates_by_direction.push_back(block_logical_coordinates[j][i]);
    }
    unsorted_coordinates.push_back(coordinates_by_direction);
  }

  // Creates ordered_x, ordered_y, and ordered_z by first sorting
  // unsorted_coordinates x, y, and z, then passing these into
  // order_sorted_elements()
  sort(unsorted_coordinates[0].begin(), unsorted_coordinates[0].end());
  std::vector<double> ordered_x =
      order_sorted_elements(unsorted_coordinates[0]);
  std::vector<double> ordered_y = {0.0};
  std::vector<double> ordered_z = {0.0};

  if (SpatialDim > 1) {
    sort(unsorted_coordinates[1].begin(), unsorted_coordinates[1].end());
    ordered_y = order_sorted_elements(unsorted_coordinates[1]);
    if (SpatialDim == 3) {
      sort(unsorted_coordinates[2].begin(), unsorted_coordinates[2].end());
      ordered_z = order_sorted_elements(unsorted_coordinates[2]);
    }
  }

  return build_connectivity_by_hexahedron<SpatialDim>(ordered_x, ordered_y,
                                                      ordered_z, block_number);
}
}  // namespace

namespace h5::detail {

// Write new connectivity connections given a std::vector of observation ids
template <size_t SpatialDim>
std::vector<int> extend_connectivity(
    std::vector<std::string>& grid_names,
    std::vector<std::vector<Spectral::Basis>>& bases,
    std::vector<std::vector<Spectral::Quadrature>>& quadratures,
    std::vector<std::vector<size_t>>& extents) {
  auto [number_of_blocks, block_number_for_each_element,
        sorted_element_indices] = compute_and_organize_block_info(grid_names);

  const auto sorted_grid_names =
      sort_by_block(sorted_element_indices, grid_names);
  const auto sorted_extents = sort_by_block(sorted_element_indices, extents);

  size_t total_expected_connectivity = 0;
  std::vector<int> expected_grid_points_per_block;
  expected_grid_points_per_block.reserve(number_of_blocks);
  std::vector<std::array<int, SpatialDim>> h_ref_per_block;
  h_ref_per_block.reserve(number_of_blocks);

  // Loop over blocks
  for (size_t j = 0; j < number_of_blocks; ++j) {
    auto [expected_connectivity_length, expected_number_of_grid_points,
          h_ref_array] =
        compute_block_level_properties<SpatialDim>(sorted_grid_names[j],
                                                   sorted_extents[j]);
    total_expected_connectivity += expected_connectivity_length;
    expected_grid_points_per_block.push_back(expected_number_of_grid_points);
    h_ref_per_block.push_back(h_ref_array);
  }

  // Create an unordered_map to be used to associate a grid point's block
  // number and coordinates as an array to the its label
  // (B#, grid_point_coord_array) -> grid_point_number
  std::unordered_map<
      std::pair<size_t, std::array<double, SpatialDim>>, size_t,
      boost::hash<std::pair<size_t, std::array<double, SpatialDim>>>>
      block_and_grid_point_map;

  // Create the sorted container for the grid points, which is a
  // std::vector<std::vector>. The length of the first layer has length equal
  // to the number of blocks, as each subvector corresponds to one of the
  // blocks. Each subvector is of length equal to the number of grid points
  // in the corresponding block, as we are storing an array of the block
  // logical coordinates for each grid point.
  std::vector<std::vector<std::array<double, SpatialDim>>>
      block_logical_coordinates_by_block;
  block_logical_coordinates_by_block.reserve(number_of_blocks);

  // Reserve size for the subvectors
  for (const auto& sorted_block_index : sorted_element_indices) {
    std::vector<std::array<double, SpatialDim>> sizing_vector;
    sizing_vector.reserve(sorted_block_index.size());
    block_logical_coordinates_by_block.push_back(sizing_vector);
  }

  // Counter for the grid points when filling the unordered_map. Grid points
  // are labelled by positive integers, so we are numbering them with this
  // counter as we associate them to the key (B#, grid_point_coord_array) in
  // the unordered map.
  size_t grid_point_number = 0;

  for (size_t element_index = 0;
       element_index < block_number_for_each_element.size(); ++element_index) {
    auto element_mesh = mesh_for_grid<SpatialDim>(
        grid_names[element_index], grid_names, extents, bases, quadratures);
    auto element_logical_coordinates_tensor = logical_coordinates(element_mesh);

    std::vector<std::array<double, SpatialDim>> element_logical_coordinates;
    element_logical_coordinates.reserve(
        element_logical_coordinates_tensor.get(0).size());

    for (size_t k = 0; k < element_logical_coordinates_tensor.get(0).size();
         ++k) {
      std::array<double, SpatialDim> logical_coords_element_increment = {};
      for (size_t l = 0; l < SpatialDim; l++) {
        gsl::at(logical_coords_element_increment, l) =
            element_logical_coordinates_tensor.get(l)[k];
      }
      element_logical_coordinates.push_back(logical_coords_element_increment);
    }

    auto block_logical_coordinates =
        generate_block_logical_coordinates<SpatialDim>(
            element_logical_coordinates, grid_names[element_index],
            h_ref_per_block[block_number_for_each_element[element_index]]);

    // Stores (B#, grid_point_coord_array) -> grid_point_number in an
    // unordered_map and grid_point_coord_array by block
    for (size_t k = 0; k < block_logical_coordinates.size(); ++k) {
      std::pair<size_t, std::array<double, SpatialDim>> block_and_grid_point(
          block_number_for_each_element[element_index],
          block_logical_coordinates[k]);
      block_and_grid_point_map.insert(
          std::pair<std::pair<size_t, std::array<double, SpatialDim>>, size_t>(
              block_and_grid_point, grid_point_number));
      grid_point_number += 1;

      block_logical_coordinates_by_block
          [block_number_for_each_element[element_index]]
              .push_back(block_logical_coordinates[k]);
    }
  }

  std::vector<int> new_connectivity;
  new_connectivity.reserve(total_expected_connectivity);

  for (size_t j = 0; j < block_logical_coordinates_by_block.size(); ++j) {
    auto block_number = j;
    auto connectivity_of_keys = generate_new_connectivity<SpatialDim>(
        block_logical_coordinates_by_block[j], block_number);
    for (const std::pair<size_t, std::array<double, SpatialDim>>& it :
         connectivity_of_keys) {
      new_connectivity.push_back(block_and_grid_point_map[it]);
    }
  }

  return new_connectivity;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                            \
  template std::vector<int> h5::detail::extend_connectivity<DIM(data)>( \
      std::vector<std::string> & grid_names,                            \
      std::vector<std::vector<Spectral::Basis>> & bases,                \
      std::vector<std::vector<Spectral::Quadrature>> & quadratures,     \
      std::vector<std::vector<size_t>> & extents);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM

}  // namespace h5::detail
