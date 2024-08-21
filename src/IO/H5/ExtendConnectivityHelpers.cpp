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
#include "Domain/Structure/SegmentId.hpp"
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

// NEW FUNCTIONS FOR NEW UPCOMING VERSION OF EXTEND CONNECTIVITY

template <size_t SpatialDim>
std::pair<std::array<size_t, SpatialDim>, std::array<size_t, SpatialDim>>
compute_index_and_refinement_for_element(const std::string& element_grid_name) {
  // Computes the refinements and indieces for an element
  // when given the grid names for that particular element. This function CANNOT
  // be given all the gridnames across the whole domain, only for the one
  // element. Returns a std::pair where the first entry contains the all the
  // indices for the element and the second entry contains the refinement in
  // every dimension of the element.

  std::array<size_t, SpatialDim> indices_of_element = {};
  std::array<size_t, SpatialDim> h_ref_of_element = {};

  // some string indexing gymnastics
  size_t grid_points_previous_start_position = 0;
  size_t grid_points_start_position = 0;
  size_t grid_points_end_position = 0;

  size_t h_ref_previous_start_position = 0;
  size_t h_ref_start_position = 0;
  size_t h_ref_end_position = 0;

  for (size_t j = 0; j < SpatialDim; ++j) {
    grid_points_start_position =
        element_grid_name.find('I', grid_points_previous_start_position + 1);
    if (j == SpatialDim - 1) {
      grid_points_end_position =
          element_grid_name.find(')', grid_points_start_position);
    } else {
      grid_points_end_position =
          element_grid_name.find(',', grid_points_start_position);
    }

    std::stringstream element_index_substring(element_grid_name.substr(
        grid_points_start_position + 1,
        grid_points_end_position - grid_points_start_position - 1));
    size_t current_element_index = 0;
    element_index_substring >> current_element_index;
    gsl::at(indices_of_element, j) = current_element_index;
    grid_points_previous_start_position = grid_points_start_position;

    h_ref_start_position =
        element_grid_name.find('L', h_ref_previous_start_position + 1);
    h_ref_end_position = element_grid_name.find('I', h_ref_start_position);
    std::stringstream element_grid_name_substring(element_grid_name.substr(
        h_ref_start_position + 1,
        h_ref_end_position - h_ref_start_position - 1));
    size_t current_element_h_ref = 0;
    element_grid_name_substring >> current_element_h_ref;
    gsl::at(h_ref_of_element, j) = current_element_h_ref;
    h_ref_previous_start_position = h_ref_start_position;
  }

  // pushes back the all indices for an element to the "indices" vector.
  return std::pair{indices_of_element, h_ref_of_element};
}

// std::pair<std::vector<std::array<size_t, SpatialDim>>,
//           std::vector<std::array<size_t, SpatialDim>>>
template <size_t SpatialDim>
std::vector<
    std::pair<std::array<size_t, SpatialDim>, std::array<size_t, SpatialDim>>>
compute_indices_and_refinements_for_elements(
    const std::vector<std::string>& block_grid_names) {
  // Computes the refinements and indieces for all the elements in a given block
  // when given the grid names for that particular block. This function CANNOT
  // be given all the gridnames across the whole domain, only for the one block.
  // Returns a std::pair where the first entry contains the all the indices for
  // all elements in a block and the second entry contains all the refinements
  // in every dimension of every element within the block.

  std::vector<
      std::pair<std::array<size_t, SpatialDim>, std::array<size_t, SpatialDim>>>
      indices_and_h_refs = {};

  // std::vector<std::array<size_t, SpatialDim>> indices = {};
  // std::vector<std::array<size_t, SpatialDim>> h_ref = {};

  for (const auto& element_grid_name : block_grid_names) {
    // pushes back the all indices for an element to the "indices" vector.
    indices_and_h_refs.push_back(
        compute_index_and_refinement_for_element<SpatialDim>(
            element_grid_name));
  }

  return indices_and_h_refs;
}

template <size_t SpatialDim>
std::vector<std::array<SegmentId, SpatialDim>> compute_block_segment_ids(
    const std::vector<std::pair<std::array<size_t, SpatialDim>,
                                std::array<size_t, SpatialDim>>>&
        indices_and_refinements_for_elements) {
  // Creates a std::vector of the segmentIds of each element in the block that
  // is the same block as the block for which the refinement and indices are
  // calculated.
  std::vector<std::array<SegmentId, SpatialDim>> block_segment_ids = {};
  std::array<SegmentId, SpatialDim> segment_ids_of_current_element{};
  for (size_t i = 0; i < indices_and_refinements_for_elements.size(); ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      SegmentId current_segment_id(
          indices_and_refinements_for_elements[i].second[j],
          indices_and_refinements_for_elements[i].first[j]);
      gsl::at(segment_ids_of_current_element, j) = current_segment_id;
    }
    block_segment_ids.push_back(segment_ids_of_current_element);
  }

  return block_segment_ids;
}

template <size_t SpatialDim>
std::vector<std::array<double, SpatialDim>> element_logical_coordinates(
    const Mesh<SpatialDim>& element_mesh) {
  // Computes the ELCs of one element. Only needs to take in the element mesh of
  // that one element

  std::array<double, SpatialDim> grid_point_coordinates{};
  std::vector<std::array<double, SpatialDim>> element_logical_coordinates = {};
  const auto element_logical_coordinates_tensor =
      logical_coordinates(element_mesh);
  for (size_t i = 0; i < element_logical_coordinates_tensor.get(0).size();
       ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      gsl::at(grid_point_coordinates, j) =
          element_logical_coordinates_tensor.get(j)[i];
    }

    element_logical_coordinates.push_back(grid_point_coordinates);
  }

  return element_logical_coordinates;
}

bool share_endpoints(const SegmentId& segment_id_1,
                     const SegmentId& segment_id_2) {
  // Returns true if segment_id_1 and segment_id_2 touch in any way (including
  // overlapping). Otherwise returns false
  const double upper_1 = segment_id_1.endpoint(Side::Upper);
  const double lower_1 = segment_id_1.endpoint(Side::Lower);
  const double upper_2 = segment_id_2.endpoint(Side::Upper);
  const double lower_2 = segment_id_2.endpoint(Side::Lower);

  if (upper_1 == upper_2 || upper_1 == lower_2 || lower_1 == upper_2 ||
      lower_1 == lower_2) {
    return true;
  }

  return false;
}

template <size_t SpatialDim>
std::vector<std::array<SegmentId, SpatialDim>> find_neighbors(
    const std::array<SegmentId, SpatialDim>& element_of_interest,
    std::vector<std::array<SegmentId, SpatialDim>>& all_elements) {
  // Identifies all neighbors(face, edge, and corner) in one big vector.
  // This does not differentiate between types of neighbors, just that they are
  // neighbors. Takes in the element of interest and the rest of the elements in
  // the block in the form of SegmentIds. This function also identifies the
  // element of interest as it's own neighbor. This is removed in
  // "neighbor_direction".

  for (size_t i = 0; i < SpatialDim; ++i) {
    const SegmentId current_segment_id = gsl::at(element_of_interest, i);
    // std::cout << "size: " << all_elements.size() << '\n';
    // lambda identifies the indicies of the non neighbors and stores it in
    // not_neighbors
    const auto not_neighbors = std::remove_if(
        all_elements.begin(), all_elements.end(),
        [&i, &current_segment_id](
            std::array<SegmentId, SpatialDim> element_to_compare) {
          // only need touches since if they overlap, they also touch
          bool touches = share_endpoints(current_segment_id,
                                         gsl::at(element_to_compare, i));
          // Filters out neighbors with higher refinement
          if (current_segment_id.refinement_level() >
              gsl::at(element_to_compare, i).refinement_level()) {
            touches = false;
          }
          return !touches;
        });
    // removes all std::array<SegmentId, SpatialDim> that aren't neighbors
    all_elements.erase(not_neighbors, all_elements.end());
  }

  // std::cout << "size: " << all_elements.size() << '\n';
  return all_elements;
}

template <size_t SpatialDim>
std::array<int, SpatialDim> neighbor_direction(
    const std::array<SegmentId, SpatialDim>& element_of_interest,
    const std::array<SegmentId, SpatialDim>& element_to_compare) {
  // identifies the direction vector of the element to compare relative to the
  // element of interest. It is known that element_to_compare is a neighbor

  std::array<int, SpatialDim> direction{};
  for (size_t i = 0; i < SpatialDim; ++i) {
    if (gsl::at(element_of_interest, i).endpoint(Side::Upper) ==
        gsl::at(element_to_compare, i).endpoint(Side::Lower)) {
      gsl::at(direction, i) = 1;
    }
    if (gsl::at(element_of_interest, i).endpoint(Side::Lower) ==
        gsl::at(element_to_compare, i).endpoint(Side::Upper)) {
      gsl::at(direction, i) = -1;
    }
    if (gsl::at(element_of_interest, i).endpoint(Side::Upper) ==
            gsl::at(element_to_compare, i).endpoint(Side::Upper) ||
        gsl::at(element_of_interest, i).endpoint(Side::Lower) ==
            gsl::at(element_to_compare, i).endpoint(Side::Lower)) {
      gsl::at(direction, i) = 0;
    }
  }

  return direction;
}

template <size_t SpatialDim>
std::vector<
    std::pair<std::array<SegmentId, SpatialDim>, std::array<int, SpatialDim>>>
compute_neighbors_with_direction(
    const std::array<SegmentId, SpatialDim>& element_of_interest,
    const std::vector<std::array<SegmentId, SpatialDim>>& all_neighbors) {
  // outputs a std::array of length SpatialDim of element segmentIds. First are
  // face neighbors, then edge neighbors, then corner neighbors. Then it also
  // returns a std::array of the direction vector of each neighbor as well in a
  // std::pair. Takes in the element of interest and the list of all neighbors.
  std::vector<
      std::pair<std::array<SegmentId, SpatialDim>, std::array<int, SpatialDim>>>
      neighbors_with_direction = {};

  for (const auto& neighbor : all_neighbors) {
    std::array<int, SpatialDim> direction =
        neighbor_direction(element_of_interest, neighbor);
    std::pair<std::array<SegmentId, SpatialDim>, std::array<int, SpatialDim>>
        neighbor_with_direction{neighbor, direction};
    int direction_magnitude = 0;

    // Why are we doing this loop for every neighbour just to be able to
    // eliminate itself
    for (size_t j = 0; j < SpatialDim; ++j) {
      direction_magnitude += abs(gsl::at(direction, j));
    }
    // Filter out itself fromt the neighbors
    if (direction_magnitude != 0) {
      neighbors_with_direction.push_back(neighbor_with_direction);
    }
  }

  return neighbors_with_direction;
}

template <size_t SpatialDim>
std::vector<std::array<double, SpatialDim>>
block_logical_coordinates_for_element(
    const std::vector<std::array<double, SpatialDim>>&
        element_logical_coordinates,
    const std::pair<std::array<size_t, SpatialDim>,
                    std::array<size_t, SpatialDim>>&
        element_indices_and_refinements,
    const bool& sort_flag) {
  // Generates the BLC of a particular element given that elements ELC and it's
  // indices and refinements as defined from its grid name.

  std::vector<std::array<double, SpatialDim>> BLCs_for_element = {};
  for (const auto& grid_point : element_logical_coordinates) {
    std::array<double, SpatialDim> BLCs_of_gridpoint{};
    for (size_t j = 0; j < SpatialDim; ++j) {
      int number_of_elements =
          two_to_the(gsl::at(element_indices_and_refinements.second, j));
      double shift = -1 + (2 * static_cast<double>(gsl::at(
                                   element_indices_and_refinements.first, j)) +
                           1) /
                              static_cast<double>(number_of_elements);
      gsl::at(BLCs_of_gridpoint, j) =
          gsl::at(grid_point, j) / number_of_elements + shift;
    }
    BLCs_for_element.push_back(BLCs_of_gridpoint);
  }

  // sort the BLC by increasing z, then y, then x
  if (sort_flag) {
    std::sort(BLCs_for_element.begin(), BLCs_for_element.end(),
              [](const auto& lhs, const auto& rhs) {
                return std::lexicographical_compare(lhs.begin(), lhs.end(),
                                                    rhs.begin(), rhs.end());
              });
  }

  return BLCs_for_element;
}

template <size_t SpatialDim>
std::string grid_name_reconstruction(
    const std::array<SegmentId, SpatialDim>& element,
    const std::vector<std::string>& block_grid_names) {
  // Gets the block number. This can be CHANGED to be passed in as an argument
  // later.
  std::string element_grid_name =
      block_grid_names[0].substr(0, block_grid_names[0].find('(', 0) + 1);
  for (size_t i = 0; i < SpatialDim; ++i) {
    element_grid_name +=
        "L" + std::to_string(gsl::at(element, i).refinement_level()) + "I" +
        std::to_string(gsl::at(element, i).index());
    if (i < SpatialDim - 1) {
      element_grid_name += ",";
    } else if (i == SpatialDim - 1) {
      element_grid_name += ")]";
    }
  }
  return element_grid_name;
}

template <size_t SpatialDim>
std::vector<std::array<double, SpatialDim>> compute_element_BLCs(
    const std::array<SegmentId, SpatialDim>& element,
    const std::vector<std::string>& block_grid_names,
    const std::vector<std::vector<size_t>>& block_extents,
    const std::vector<std::vector<Spectral::Basis>>& block_bases,
    const std::vector<std::vector<Spectral::Quadrature>>& block_quadratures,
    const std::vector<std::pair<std::array<size_t, SpatialDim>,
                                std::array<size_t, SpatialDim>>>&
        indices_and_refinements_for_elements) {
  // grid_name_reconstruction requires block_grid_names for the block number.
  // This should be CHANGED!!! later when we loop over the blocks to take in
  // the index of the block.

  const std::string element_grid_name =
      grid_name_reconstruction<SpatialDim>(element, block_grid_names);
  // Find the index of the element of interest within grid_names so we can
  // find it later in extents, bases, and quadratures
  const auto found_element_grid_name =
      alg::find(block_grid_names, element_grid_name);
  const auto element_index = static_cast<size_t>(
      std::distance(block_grid_names.begin(), found_element_grid_name));

  // Construct the mesh for the element of interest. mesh_for_grid finds the
  // index internally for us.
  const Mesh<SpatialDim> element_mesh = h5::mesh_for_grid<SpatialDim>(
      element_grid_name, block_grid_names, block_extents, block_bases,
      block_quadratures);

  // Compute the element logical coordinates for the element of interest
  std::vector<std::array<double, SpatialDim>> element_ELCs =
      element_logical_coordinates<SpatialDim>(element_mesh);

  // Access the indices and refinements for the element of interest and change
  // container type. Was orginally a std::pair<std::vector<...>,
  // std::vector<...>>. We index into the vectors and reconstruct the pair.
  const std::pair<std::array<size_t, SpatialDim>,
                  std::array<size_t, SpatialDim>>&
      element_indices_and_refinements{
          indices_and_refinements_for_elements[element_index].first,
          indices_and_refinements_for_elements[element_index].second};

  // Compute BLC for the element of interest
  const std::vector<std::array<double, SpatialDim>>& element_BLCs =
      block_logical_coordinates_for_element<SpatialDim>(
          element_ELCs, element_indices_and_refinements, true);

  return element_BLCs;
}

// Need offsets to determine how to index element for gridpoints
template <size_t SpatialDim>
std::pair<size_t, size_t> gridpoints_BLCs_dim_offsets(
    const std::vector<std::array<double, SpatialDim>>&
        element_gridpoints_BLCs) {
  // Comparison function to check if BLC at index dimension is equal
  auto is_equal = [element_gridpoints_BLCs](
                      std::array<double, SpatialDim> gridpoint_BLCs,
                      size_t index) {
    return gridpoint_BLCs[index] == element_gridpoints_BLCs[0][index];
  };

  // Find first value at which is_equal is not true in both dimensions
  // Add check that iterator did not return last, i.e. couldn't determine offset
  size_t y_offset_index = std::distance(
      element_gridpoints_BLCs.begin(),
      std::find_if_not(element_gridpoints_BLCs.begin(),
                       element_gridpoints_BLCs.end(),
                       std::bind(is_equal, std::placeholders::_1, 1)));
  size_t x_offset_index = std::distance(
      element_gridpoints_BLCs.begin(),
      std::find_if_not(element_gridpoints_BLCs.begin(),
                       element_gridpoints_BLCs.end(),
                       std::bind(is_equal, std::placeholders::_1, 0)));

  return std::make_pair(x_offset_index, y_offset_index);
}

// Returns the BLCs and directions of all the neighbors in
// neighbors_with_direction
template <size_t SpatialDim>
std::vector<std::pair<std::vector<std::array<double, SpatialDim>>,
                      std::array<int, SpatialDim>>>
compute_neighbor_BLCs_and_directions(
    const std::vector<std::string>& block_grid_names,
    const std::vector<std::vector<size_t>>& block_extents,
    const std::vector<std::vector<Spectral::Basis>>& block_bases,
    const std::vector<std::vector<Spectral::Quadrature>>& block_quadratures,
    const std::vector<std::pair<std::array<SegmentId, SpatialDim>,
                                std::array<int, SpatialDim>>>&
        neighbors_with_direction,
    const std::vector<std::pair<std::array<size_t, SpatialDim>,
                                std::array<size_t, SpatialDim>>>&
        indices_and_refinements_for_elements) {
  // Will store all neighbor BLC's and their direction vectors
  std::vector<std::pair<std::vector<std::array<double, SpatialDim>>,
                        std::array<int, SpatialDim>>>
      neighbor_BLCs_and_directions = {};
  // Need to loop over all the neighbors. First by neighbor type then the
  // number of neighbor in each type.
  for (size_t k = 0; k < neighbors_with_direction.size(); ++k) {
    // Reconstruct the grid name for the neighboring element
    // std::cout << "Neighbor grid name: " << '\n';
    // std::cout << "Neighbor BLCs: " << '\n';
    const std::vector<std::array<double, SpatialDim>> neighbor_BLCs =
        compute_element_BLCs(neighbors_with_direction[k].first,
                             block_grid_names, block_extents, block_bases,
                             block_quadratures,
                             indices_and_refinements_for_elements);

    // Access the direction vector. Lives inside neighbors_with_direction
    // datastructure
    const std::array<int, SpatialDim> neighbor_direction =
        gsl::at(neighbors_with_direction, k).second;
    // std::cout << "direction vector: " <<
    // neighbors_with_direction[k].second[0]
    //           << ", " << neighbors_with_direction[k].second[1] << ", "
    //           << neighbors_with_direction[k].second[2] << '\n';

    std::pair<std::vector<std::array<double, SpatialDim>>,
              std::array<int, SpatialDim>>
        neighbor_BLCs_and_direction{neighbor_BLCs, neighbor_direction};
    neighbor_BLCs_and_directions.push_back(neighbor_BLCs_and_direction);
  }

  return neighbor_BLCs_and_directions;
}

template <size_t SpatialDim>
std::vector<std::pair<std::vector<std::array<double, SpatialDim>>,
                      std::array<int, SpatialDim>>>
compute_neighbor_info(
    const std::array<SegmentId, SpatialDim>& element_of_interest,
    std::vector<std::array<SegmentId, SpatialDim>>& neighbor_segment_ids,
    const std::vector<std::string>& block_grid_names,
    const std::vector<std::vector<size_t>>& block_extents,
    const std::vector<std::vector<Spectral::Basis>>& block_bases,
    const std::vector<std::vector<Spectral::Quadrature>>& block_quadratures,
    const std::vector<std::pair<std::array<size_t, SpatialDim>,
                                std::array<size_t, SpatialDim>>>&
        indices_and_refinements_for_elements) {
  const std::vector<std::array<SegmentId, SpatialDim>> all_neighbors =
      find_neighbors(element_of_interest, neighbor_segment_ids);

  // Gives all neighbors sorted by type. First is face, then edge, then
  // corner. Also gives the direction vector in the pair.
  const std::vector<
      std::pair<std::array<SegmentId, SpatialDim>, std::array<int, SpatialDim>>>
      neighbors_with_direction =
          compute_neighbors_with_direction(element_of_interest, all_neighbors);

  // Stores all neighbor BLCs and directions in neighbor_BLCs_and_directions
  const std::vector<std::pair<std::vector<std::array<double, SpatialDim>>,
                              std::array<int, SpatialDim>>>
      neighbor_BLCs_and_directions = compute_neighbor_BLCs_and_directions(
          block_grid_names, block_extents, block_bases, block_quadratures,
          neighbors_with_direction, indices_and_refinements_for_elements);

  return neighbor_BLCs_and_directions;
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

// NEW MAIN EXTEND CONNECTIVITY FUNCTIONS

// Write new connectivity connections given a std::vector of observation ids
template <size_t SpatialDim>
std::vector<std::array<double, SpatialDim>> extend_connectivity_by_block(
    const std::vector<std::string>& block_grid_names,
    const std::vector<std::vector<size_t>>& block_extents,
    const std::vector<std::vector<Spectral::Basis>>& block_bases,
    const std::vector<std::vector<Spectral::Quadrature>>& block_quadratures) {
  std::vector<std::array<double, SpatialDim>> block_connectivity;

  // std::pair of the indices(first entry) and refinements (second entry) for
  // the entire block.
  // const std::pair<std::vector<std::array<size_t, SpatialDim>>,
  //                 std::vector<std::array<size_t, SpatialDim>>>&
  const std::vector<std::pair<std::array<size_t, SpatialDim>,
                              std::array<size_t, SpatialDim>>>&
      indices_and_refinements_for_elements =
          compute_indices_and_refinements_for_elements<SpatialDim>(
              block_grid_names);

  // Segment Ids for every element in the block. Each element has SpatialDim
  // SegmentIds, one for each dimension.
  const std::vector<std::array<SegmentId, SpatialDim>>& block_segment_ids =
      compute_block_segment_ids<SpatialDim>(
          indices_and_refinements_for_elements);

  for (size_t i = 0; i < block_segment_ids.size(); ++i) {
    // Need to copy block_segment_ids to a new container since it will be
    // altered.
    std::vector<std::array<SegmentId, SpatialDim>> neighbor_segment_ids =
        block_segment_ids;
    // Identify the element I want to find the neighbors of.
    const std::array<SegmentId, SpatialDim>& element_of_interest =
        gsl::at(block_segment_ids, i);

    // Need the grid name of the element of interest to reverse search it in the
    // vector of all grid names to get its position in that vector
    // std::cout << "Element of interest grid name: " << '\n';
    // std::cout << "Element of interest BLCs: " << '\n';
    std::vector<std::array<double, SpatialDim>> element_of_interest_BLCs =
        compute_element_BLCs(element_of_interest, block_grid_names,
                             block_extents, block_bases, block_quadratures,
                             indices_and_refinements_for_elements);

    std::pair<size_t, size_t> elem_of_int_dim_offsets =
        gridpoints_BLCs_dim_offsets(element_of_interest_BLCs);

    // Stores all neighbor BLCs and directions in neighbor_info
    std::vector<std::pair<std::vector<std::array<double, SpatialDim>>,
                          std::array<int, SpatialDim>>>
        neighbour_info = compute_neighbor_info<SpatialDim>(
            element_of_interest, neighbor_segment_ids, block_grid_names,
            block_extents, block_bases, block_quadratures,
            indices_and_refinements_for_elements);

    /*  To be added: Determining the connectivity between the element of
        interest and the neighbour
    */
  }

  return block_connectivity;
}

/*  Write new connectivity connections given the grid_names, bases, quadratures,
    and extents
*/
template <size_t SpatialDim>
std::vector<int> new_extend_connectivity(
    std::vector<std::string>& grid_names,
    std::vector<std::vector<Spectral::Basis>>& bases,
    std::vector<std::vector<Spectral::Quadrature>>& quadratures,
    std::vector<std::vector<size_t>>& extents) {
  auto [number_of_blocks, block_number_for_each_element,
        sorted_element_indices] = compute_and_organize_block_info(grid_names);

  const auto sorted_grid_names =
      sort_by_block(sorted_element_indices, grid_names);
  const auto sorted_extents = sort_by_block(sorted_element_indices, extents);
  const auto sorted_bases = sort_by_block(sorted_element_indices, bases);
  const auto sorted_quadratures =
      sort_by_block(sorted_element_indices, quadratures);

  // Create an unordered_map to be used to associate a grid point's block
  // number and coordinates as an array to the its label
  // (B#, grid_point_coord_array) -> grid_point_number
  std::unordered_map<
      std::pair<size_t, std::array<double, SpatialDim>>, size_t,
      boost::hash<std::pair<size_t, std::array<double, SpatialDim>>>>
      block_and_grid_point_map;

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

    const std::pair<std::array<size_t, SpatialDim>,
                    std::array<size_t, SpatialDim>>& element_inds_and_refs =
        compute_index_and_refinement_for_element<SpatialDim>(
            grid_names[element_index]);

    std::vector<std::array<double, SpatialDim>> block_logical_coordinates =
        block_logical_coordinates_for_element<SpatialDim>(
            element_logical_coordinates, element_inds_and_refs, false);

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
    }
  }

  std::vector<int> new_connectivity;

  /*  Formula for extended connectivity size assumes face, edge, and corner
      neighbours are being connected. Since only face neighbours is complete as
      of now, reserving memory for the connectivity vector is skipped for now.
  */

  // size_t total_expected_connectivity = 0;
  // new_connectivity.reserve(total_expected_connectivity);

  for (size_t j = 0; j < sorted_element_indices.size(); ++j) {
    std::vector<std::array<double, SpatialDim>> block_connectivity =
        extend_connectivity_by_block<SpatialDim>(
            sorted_grid_names[j], sorted_extents[j], sorted_bases[j],
            sorted_quadratures[j]);

    for (const std::array<double, SpatialDim>& it : block_connectivity) {
      new_connectivity.push_back(
          block_and_grid_point_map[std::make_pair(j, it)]);
    }
  }

  return new_connectivity;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                \
  template std::vector<int> h5::detail::extend_connectivity<DIM(data)>(     \
      std::vector<std::string> & grid_names,                                \
      std::vector<std::vector<Spectral::Basis>> & bases,                    \
      std::vector<std::vector<Spectral::Quadrature>> & quadratures,         \
      std::vector<std::vector<size_t>> & extents);                          \
  template std::vector<int> h5::detail::new_extend_connectivity<DIM(data)>( \
      std::vector<std::string> & grid_names,                                \
      std::vector<std::vector<Spectral::Basis>> & bases,                    \
      std::vector<std::vector<Spectral::Quadrature>> & quadratures,         \
      std::vector<std::vector<size_t>> & extents);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM

}  // namespace h5::detail
