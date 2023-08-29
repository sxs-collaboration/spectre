// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/Domain/DomainTestHelpers.hpp"

#include <algorithm>
#include <cmath>
#include <ios>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Domain/Block.hpp"                  // IWYU pragma: keep
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"  // IWYU pragma: keep
#include "Domain/DomainHelpers.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"  // IWYU pragma: keep
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Structure/Neighbors.hpp"  // IWYU pragma: keep
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

template <size_t VolumeDim>
class Element;

namespace domain {
namespace {
template <size_t VolumeDim>
boost::rational<size_t> fraction_of_block_face_area(
    const ElementId<VolumeDim>& element_id,
    const Direction<VolumeDim>& direction) {
  const auto segment_ids = element_id.segment_ids();
  size_t sum_of_refinement_levels = 0;
  const size_t dim_normal_to_face = direction.dimension();
  for (size_t d = 0; d < VolumeDim; ++d) {
    if (dim_normal_to_face != d) {
      sum_of_refinement_levels += gsl::at(segment_ids, d).refinement_level();
    }
  }
  return {1, two_to_the(sum_of_refinement_levels)};
}

template <size_t VolumeDim>
bool is_a_neighbor_of_my_neighbor_me(
    const ElementId<VolumeDim>& my_id, const Element<VolumeDim>& my_neighbor,
    const Direction<VolumeDim>& direction_to_me_in_neighbor) {
  size_t number_of_matches = 0;
  for (const auto& id_of_neighbor_of_neighbor :
       my_neighbor.neighbors().at(direction_to_me_in_neighbor).ids()) {
    if (my_id == id_of_neighbor_of_neighbor) {
      ++number_of_matches;
    }
  }
  return 1 == number_of_matches;
}

template <size_t VolumeDim>
void test_domain_connectivity(
    const Domain<VolumeDim>& domain,
    const std::unordered_map<ElementId<VolumeDim>, Element<VolumeDim>>&
        elements_in_domain) {
  boost::rational<size_t> volume_of_elements{0};
  boost::rational<size_t> surface_area_of_external_boundaries{0};
  // For internal boundaries within a Block, lower/upper is defined with
  // respect to the logical coordinates of the block
  // for internal boundaries between two Blocks, lower/upper is defined
  // using the id of the Block (the Block with the smaller/larger id being
  // associated with the lower/upper internal boundary).
  boost::rational<size_t> surface_area_of_lower_internal_boundaries{0};
  boost::rational<size_t> surface_area_of_upper_internal_boundaries{0};

  for (const auto& key_value : elements_in_domain) {
    const auto& element_id = key_value.first;
    const auto& element = key_value.second;
    CHECK(element_id == element.id());
    volume_of_elements += fraction_of_block_volume(element_id);
    for (const auto& direction : element.external_boundaries()) {
      surface_area_of_external_boundaries +=
          fraction_of_block_face_area(element_id, direction);
    }
    for (const auto& direction_neighbors : element.neighbors()) {
      const auto& direction = direction_neighbors.first;
      const auto& neighbors = direction_neighbors.second;
      const auto& direction_to_me_in_neighbor =
          neighbors.orientation()(direction.opposite());
      for (const auto& neighbor_id : neighbors.ids()) {
        CHECK(is_a_neighbor_of_my_neighbor_me(
            element_id, elements_in_domain.at(neighbor_id),
            direction_to_me_in_neighbor));
      }
      const size_t element_block_id = element_id.block_id();
      const size_t neighbor_block_id = neighbors.ids().begin()->block_id();
      if (element_block_id == neighbor_block_id) {
        if (Side::Lower == direction.side()) {
          surface_area_of_lower_internal_boundaries +=
              fraction_of_block_face_area(element_id, direction);
        } else {
          surface_area_of_upper_internal_boundaries +=
              fraction_of_block_face_area(element_id, direction);
        }
      } else {
        if (neighbor_block_id > element_block_id) {
          surface_area_of_lower_internal_boundaries +=
              fraction_of_block_face_area(element_id, direction);
        } else {
          surface_area_of_upper_internal_boundaries +=
              fraction_of_block_face_area(element_id, direction);
        }
      }
    }
  }

  CHECK(boost::rational<size_t>{domain.blocks().size()} == volume_of_elements);
  CHECK(surface_area_of_lower_internal_boundaries ==
        surface_area_of_upper_internal_boundaries);
  size_t number_of_external_block_faces{0};
  for (const auto& block : domain.blocks()) {
    number_of_external_block_faces += block.external_boundaries().size();
  }
  CHECK(boost::rational<size_t>{number_of_external_block_faces} ==
        surface_area_of_external_boundaries);
}

template <size_t AllowedDifference>
SPECTRE_ALWAYS_INLINE void check_if_levels_are_within(const size_t level_1,
                                                      const size_t level_2) {
  // clang-tidy complains about a catch-internal do {} while (false) loop.
  // NOLINTNEXTLINE(bugprone-infinite-loop)
  CHECK(level_1 <= level_2 + AllowedDifference);
  // NOLINTNEXTLINE(bugprone-infinite-loop)
  CHECK(level_2 <= level_1 + AllowedDifference);
}

template <size_t AllowedTangentialDifference, size_t VolumeDim>
void test_refinement_levels_of_neighbors(
    const std::unordered_map<ElementId<VolumeDim>, Element<VolumeDim>>&
        elements) {
  for (const auto& key_value : elements) {
    const auto& element_id = key_value.first;
    const auto& element = key_value.second;
    for (const auto& direction_neighbors : element.neighbors()) {
      const auto& neighbors = direction_neighbors.second;
      const auto& orientation = neighbors.orientation();
      for (size_t d = 0; d < VolumeDim; ++d) {
        // No restriction on perpendicular refinement.
        if (d == direction_neighbors.first.dimension()) {
          continue;
        }

        const size_t my_dim_in_neighbor = orientation(d);
        const size_t my_level = element_id.segment_id(d).refinement_level();
        for (const auto neighbor_id : neighbors.ids()) {
          const size_t neighbor_level =
              neighbor_id.segment_id(my_dim_in_neighbor).refinement_level();
          check_if_levels_are_within<AllowedTangentialDifference>(
              my_level, neighbor_level);
        }
      }
    }
  }
}

template <size_t VolumeDim>
bool blocks_are_neighbors(const Block<VolumeDim>& host_block,
                          const Block<VolumeDim>& neighbor_block) {
  return alg::any_of(host_block.neighbors(),
                     [&neighbor_block](const auto& neighbor) {
                       return neighbor.second.id() == neighbor_block.id();
                     });
}

// Finds the OrientationMap of a neighboring Block relative to a host Block.
template <size_t VolumeDim>
OrientationMap<VolumeDim> find_neighbor_orientation(
    const Block<VolumeDim>& host_block,
    const Block<VolumeDim>& neighbor_block) {
  for (const auto& neighbor : host_block.neighbors()) {
    if (neighbor.second.id() == neighbor_block.id()) {
      return neighbor.second.orientation();
    }
  }
  ERROR("The Block `neighbor_block` is not a neighbor of `host_block`.");
}

// Finds the Direction to the neighboring Block relative to a host Block.
template <size_t VolumeDim>
Direction<VolumeDim> find_direction_to_neighbor(
    const Block<VolumeDim>& host_block,
    const Block<VolumeDim>& neighbor_block) {
  for (const auto& neighbor : host_block.neighbors()) {
    if (neighbor.second.id() == neighbor_block.id()) {
      return neighbor.first;
    }
  }
  ERROR("The Block `neighbor_block` is not a neighbor of `host_block`.");
}

// Convert Point to Directions, for use with OrientationMap
template <size_t VolumeDim>
std::array<Direction<VolumeDim>, VolumeDim> get_orthant(
    const tnsr::I<double, VolumeDim, Frame::BlockLogical>& point) {
  std::array<Direction<VolumeDim>, VolumeDim> result;
  for (size_t i = 0; i < VolumeDim; i++) {
    gsl::at(result, i) =
        Direction<VolumeDim>(i, point[i] >= 0 ? Side::Upper : Side::Lower);
  }
  return result;
}

// Convert Directions to Point, for use with CoordinateMap
template <size_t VolumeDim>
tnsr::I<double, VolumeDim, Frame::BlockLogical> get_corner_of_orthant(
    const std::array<Direction<VolumeDim>, VolumeDim>& directions) {
  tnsr::I<double, VolumeDim, Frame::BlockLogical> result{};
  for (size_t i = 0; i < VolumeDim; i++) {
    result[gsl::at(directions, i).dimension()] =
        gsl::at(directions, i).side() == Side::Upper ? 1.0 : -1.0;
  }
  return result;
}

// The relative OrientationMap between Blocks induces a map that takes
// Points in the host Block to Points in the neighbor Block.
template <size_t VolumeDim>
tnsr::I<double, VolumeDim, Frame::BlockLogical> point_in_neighbor_frame(
    const OrientationMap<VolumeDim>& orientation,
    const tnsr::I<double, VolumeDim, Frame::BlockLogical>& point) {
  auto point_get_orthant = get_orthant(point);
  std::for_each(
      point_get_orthant.begin(), point_get_orthant.end(),
      [&orientation](auto& direction) { direction = orientation(direction); });
  return get_corner_of_orthant(point_get_orthant);
}

// This tests whether a logical grid point on the face of the host Block
// corresponds to the same logical grid point on the abutting face of the
// neighbor Block (taking into account the discrete rotation of the
// OrientationMap from the host Block to the neighbor Block).
template <size_t VolumeDim>
void check_block_face_grid_points_align(
    const Block<VolumeDim>& host_block, const Block<VolumeDim>& neighbor_block,
    double time = std::numeric_limits<double>::signaling_NaN(),
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time = {}) {
  const auto direction = find_direction_to_neighbor(host_block, neighbor_block);
  const auto orientation =
      find_neighbor_orientation(host_block, neighbor_block);
  // Set up a Mesh on the shared face. Corner points are already checked by
  // physical_separation, so check on Gauss points
  const Mesh<VolumeDim - 1> face_mesh{3_st,
                                      SpatialDiscretization::Basis::Legendre,
                                      SpatialDiscretization::Quadrature::Gauss};
  // We want block logical coordinates on face_mesh on the host side and the
  // neighbor. The following returns element logical coordinates in the host
  // frame, which we then copy into tensors holding block logical coordinates,
  // taking into account the OrientationMap between the host and neighbor
  // blocks.
  tnsr::I<DataVector, VolumeDim, Frame::ElementLogical> xi =
      interface_logical_coordinates(face_mesh, direction);
  tnsr::I<DataVector, VolumeDim, Frame::BlockLogical> xi_host{};
  tnsr::I<DataVector, VolumeDim, Frame::BlockLogical> xi_neighbor{};
  for (size_t d = 0; d < VolumeDim; ++d) {
    // assume an Element covering the Block which maps element logical
    // coordinates to block logical coordinates with the identity map
    xi_host[d] = xi[d];
    const auto dth_upper_direction_in_neighbor_frame =
        orientation(Direction<VolumeDim>(d, Side::Upper));
    // This takes into account the permutation of dimensions induced by the
    // OrientationMap
    xi_neighbor[dth_upper_direction_in_neighbor_frame.dimension()] = xi[d];
    // There is a sign flip if this is the dimension of the direction to the
    // neighbor as the logical coord is -1/+1 on the lower/upper side of the
    // block.
    // There is also a sign flip if the mapped upper direction becomes a lower
    // direction
    if ((dth_upper_direction_in_neighbor_frame.side() == Side::Lower) xor
        (d == direction.dimension())) {
      xi_neighbor[dth_upper_direction_in_neighbor_frame.dimension()] *= -1.0;
    }
  }
  if (host_block.is_time_dependent() != neighbor_block.is_time_dependent()) {
    ERROR(
        "Both host_block and neighbor_block must have the same time "
        "dependence, but host_block has time-dependence status: "
        << std::boolalpha << host_block.is_time_dependent()
        << " and neighbor_block has: " << neighbor_block.is_time_dependent());
  }
  CAPTURE(xi_host);
  CAPTURE(xi_neighbor);
  // Check that each grid point on the face Mesh has the same grid and inertial
  // coordinates when mapped from the logical coordinates of each Block.
  if (host_block.is_time_dependent()) {
    const auto& host_map_logical_to_grid =
        host_block.moving_mesh_logical_to_grid_map();
    const auto& host_map_grid_to_inertial =
        host_block.moving_mesh_grid_to_inertial_map();
    const auto& neighbor_map_logical_to_grid =
        neighbor_block.moving_mesh_logical_to_grid_map();
    const auto& neighbor_map_grid_to_inertial =
        neighbor_block.moving_mesh_grid_to_inertial_map();
    const auto x_grid_self = host_map_logical_to_grid(xi_host);
    const auto x_grid_neighbor = neighbor_map_logical_to_grid(xi_neighbor);
    CHECK_ITERABLE_APPROX(x_grid_self, x_grid_neighbor);
    const auto x_inertial_self =
        host_map_grid_to_inertial(x_grid_self, time, functions_of_time);
    const auto x_inertial_neighbor =
        neighbor_map_grid_to_inertial(x_grid_neighbor, time, functions_of_time);
    CHECK_ITERABLE_APPROX(x_inertial_self, x_inertial_neighbor);
  } else {
    const auto& host_map = host_block.stationary_map();
    const auto& neighbor_map = neighbor_block.stationary_map();
    const auto x_self = host_map(xi_host);
    const auto x_neighbor = neighbor_map(xi_neighbor);
    CHECK_ITERABLE_APPROX(x_self, x_neighbor);
  }
}

// Given two Blocks which are neighbors, computes the max separation between
// the abutting faces of the Blocks in the Frame::Inertial frame using the
// CoordinateMaps of each block.
template <size_t VolumeDim>
double physical_separation(
    const Block<VolumeDim>& block1, const Block<VolumeDim>& block2,
    double time = std::numeric_limits<double>::signaling_NaN(),
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time = {}) {
  double max_separation = 0;
  const auto direction = find_direction_to_neighbor(block1, block2);
  const auto orientation = find_neighbor_orientation(block1, block2);
  std::array<tnsr::I<double, VolumeDim, Frame::BlockLogical>,
             two_to_the(VolumeDim - 1)>
      shared_points1{};
  std::array<tnsr::I<double, VolumeDim, Frame::BlockLogical>,
             two_to_the(VolumeDim - 1)>
      shared_points2{};
  for (FaceCornerIterator<VolumeDim> fci(direction); fci; ++fci) {
    gsl::at(shared_points1, fci.face_index()) = fci();
  }
  for (FaceCornerIterator<VolumeDim> fci(direction.opposite()); fci; ++fci) {
    gsl::at(shared_points2, fci.face_index()) =
        point_in_neighbor_frame(orientation, fci());
  }
  if (block1.is_time_dependent() != block2.is_time_dependent()) {
    ERROR(
        "Both block1 and block2 must have the same time dependence, but block1 "
        "has time-dependence status: "
        << std::boolalpha << block1.is_time_dependent()
        << " and block2 has: " << block2.is_time_dependent());
  }
  if (block1.is_time_dependent()) {
    const auto& map1_logical_to_grid = block1.moving_mesh_logical_to_grid_map();
    const auto& map1_grid_to_inertial =
        block1.moving_mesh_grid_to_inertial_map();
    const auto& map2_logical_to_grid = block2.moving_mesh_logical_to_grid_map();
    const auto& map2_grid_to_inertial =
        block2.moving_mesh_grid_to_inertial_map();
    for (size_t i = 0; i < two_to_the(VolumeDim - 1); i++) {
      for (size_t j = 0; j < VolumeDim; j++) {
        max_separation = std::max(
            max_separation,
            std::abs(map1_grid_to_inertial(
                         map1_logical_to_grid(gsl::at(shared_points1, i)), time,
                         functions_of_time)
                         .get(j) -
                     map2_grid_to_inertial(
                         map2_logical_to_grid(gsl::at(shared_points2, i)), time,
                         functions_of_time)
                         .get(j)));
      }
    }
  } else {
    const auto& map1 = block1.stationary_map();
    const auto& map2 = block2.stationary_map();
    for (size_t i = 0; i < two_to_the(VolumeDim - 1); i++) {
      for (size_t j = 0; j < VolumeDim; j++) {
        max_separation = std::max(
            max_separation, std::abs(map1(gsl::at(shared_points1, i)).get(j) -
                                     map2(gsl::at(shared_points2, i)).get(j)));
      }
    }
  }
  return max_separation;
}
}  // namespace
}  // namespace domain

namespace {
template <size_t Dim>
void dispatch_check_if_maps_are_equal(
    const Block<Dim>& block,
    const domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, Dim>&
        expected_map,
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) {
  check_if_maps_are_equal(expected_map, block.stationary_map(), time,
                          functions_of_time);
}

template <size_t Dim>
void dispatch_check_if_maps_are_equal(
    const Block<Dim>& block,
    const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>&
        expected_map,
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) {
  check_if_maps_are_equal(expected_map,
                          block.moving_mesh_grid_to_inertial_map(), time,
                          functions_of_time);
}

template <size_t Dim>
void dispatch_check_if_maps_are_equal(
    const Block<Dim>& block,
    const domain::CoordinateMapBase<Frame::BlockLogical, Frame::Grid, Dim>&
        expected_map,
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) {
  check_if_maps_are_equal(expected_map, block.moving_mesh_logical_to_grid_map(),
                          time, functions_of_time);
}
}  // namespace

template <size_t VolumeDim, typename TargetFrameGridOrInertial>
void test_domain_construction(
    const Domain<VolumeDim>& domain,
    const std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<Direction<VolumeDim>>>&
        expected_external_boundaries,
    const std::vector<std::unique_ptr<domain::CoordinateMapBase<
        Frame::BlockLogical, TargetFrameGridOrInertial, VolumeDim>>>&
        expected_maps,
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>>>&
        expected_grid_to_inertial_maps) {
  const auto& blocks = domain.blocks();
  CHECK(blocks.size() == expected_external_boundaries.size());
  CHECK(blocks.size() == expected_block_neighbors.size());
  CHECK(blocks.size() == expected_maps.size());
  for (size_t i = 0; i < blocks.size(); ++i) {
    const auto& block = blocks[i];
    CHECK(block.id() == i);
    CHECK(block.neighbors() == expected_block_neighbors[i]);
    CHECK(block.external_boundaries() == expected_external_boundaries[i]);
    dispatch_check_if_maps_are_equal(block, *expected_maps[i], time,
                                     functions_of_time);
    if (std::is_same<TargetFrameGridOrInertial, Frame::Grid>::value) {
      if (expected_grid_to_inertial_maps.size() != blocks.size()) {
        ERROR("Need at least one grid to inertial map for each block ("
              << blocks.size() << " blocks in total) but received "
              << expected_grid_to_inertial_maps.size() << " instead.");
      }
      dispatch_check_if_maps_are_equal(
          block, *expected_grid_to_inertial_maps[i], time, functions_of_time);
    }
  }
}

template <size_t VolumeDim>
void test_physical_separation(
    const std::vector<Block<VolumeDim>>& blocks, double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) {
  INFO("Test physical separation");
  double tolerance = 1e-10;
  for (size_t i = 0; i < blocks.size() - 1; i++) {
    for (size_t j = i + 1; j < blocks.size(); j++) {
      if (domain::blocks_are_neighbors(blocks[i], blocks[j])) {
        CAPTURE(i);
        CAPTURE(j);
        CHECK(domain::physical_separation(blocks[i], blocks[j], time,
                                          functions_of_time) < tolerance);
        if constexpr (VolumeDim > 1) {
          domain::check_block_face_grid_points_align(blocks[i], blocks[j], time,
                                                     functions_of_time);
        }
      }
    }
  }
}

// Given a vector of Blocks, tests that the determinant of the Jacobian is
// positive at all corners of those Blocks.
template <size_t VolumeDim>
void test_det_jac_positive(
    const std::vector<Block<VolumeDim>>& blocks, double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) {
  INFO("Test determinant of Jacobian is positive");
  std::array<tnsr::I<double, VolumeDim, Frame::BlockLogical>,
             two_to_the(VolumeDim)>
      corner_points{};
  size_t index = 0;
  for (VolumeCornerIterator<VolumeDim> vci; vci; ++vci, ++index) {
    gsl::at(corner_points, index) =
        tnsr::I<double, VolumeDim, Frame::BlockLogical>(vci.coords_of_corner());
  }
  for(const auto& block: blocks) {
    CAPTURE(block.id());
    if (block.is_time_dependent()) {
      const auto& map_logical_to_grid = block.moving_mesh_logical_to_grid_map();
      const auto& map_grid_to_inertial =
          block.moving_mesh_grid_to_inertial_map();
      for (size_t i = 0; i < two_to_the(VolumeDim); i++) {
        CAPTURE(i);
        CHECK(get(determinant(map_logical_to_grid.jacobian(
                  gsl::at(corner_points, i)))) > 0.0);
        CHECK(get(determinant(map_grid_to_inertial.jacobian(
                  map_logical_to_grid(gsl::at(corner_points, i)), time,
                  functions_of_time))) > 0.0);
      }
    } else {
      const auto& map = block.stationary_map();
      for (size_t i = 0; i < two_to_the(VolumeDim); i++) {
        CAPTURE(i);
        CHECK(get(determinant(map.jacobian(gsl::at(corner_points, i)))) >
              0.0);
      }
    }
  }
}

template <size_t VolumeDim>
boost::rational<size_t> fraction_of_block_volume(
    const ElementId<VolumeDim>& element_id) {
  const auto segment_ids = element_id.segment_ids();
  size_t sum_of_refinement_levels = 0;
  for (const auto& segment_id : segment_ids) {
    sum_of_refinement_levels += segment_id.refinement_level();
  }
  return {1, two_to_the(sum_of_refinement_levels)};
}

template <size_t VolumeDim>
void test_initial_domain(const Domain<VolumeDim>& domain,
                         const std::vector<std::array<size_t, VolumeDim>>&
                             initial_refinement_levels) {
  const auto element_ids = initial_element_ids(initial_refinement_levels);
  const auto& blocks = domain.blocks();
  CHECK(blocks.size() == initial_refinement_levels.size());
  std::unordered_map<ElementId<VolumeDim>, Element<VolumeDim>> elements;
  for (const auto& element_id : element_ids) {
    elements.emplace(element_id,
                     domain::Initialization::create_initial_element(
                         element_id, blocks[element_id.block_id()],
                         initial_refinement_levels));
  }
  domain::test_domain_connectivity(domain, elements);
  domain::test_refinement_levels_of_neighbors<1>(elements);
}

template <typename DataType, size_t SpatialDim>
tnsr::i<DataType, SpatialDim> euclidean_basis_vector(
    const Direction<SpatialDim>& direction, const DataType& used_for_size) {
  auto basis_vector =
      make_with_value<tnsr::i<DataType, SpatialDim>>(used_for_size, 0.0);

  basis_vector.get(direction.axis()) =
      make_with_value<DataType>(used_for_size, direction.sign());

  return basis_vector;
}

template <typename DataType, size_t SpatialDim>
tnsr::i<DataType, SpatialDim> unit_basis_form(
    const Direction<SpatialDim>& direction,
    const tnsr::II<DataType, SpatialDim>& inv_spatial_metric) {
  auto basis_form =
      euclidean_basis_vector(direction, get<0, 0>(inv_spatial_metric));
  const DataType inv_norm =
      1.0 / get(magnitude(basis_form, inv_spatial_metric));
  for (size_t i = 0; i < SpatialDim; ++i) {
    basis_form.get(i) *= inv_norm;
  }
  return basis_form;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                          \
  template boost::rational<size_t> fraction_of_block_volume(          \
      const ElementId<DIM(data)>& element_id);                        \
  template void test_initial_domain(                                  \
      const Domain<DIM(data)>& domain,                                \
      const std::vector<std::array<size_t, DIM(data)>>&               \
          initial_refinement_levels);                                 \
  template void test_physical_separation(                             \
      const std::vector<Block<DIM(data)>>& blocks, const double time, \
      const std::unordered_map<                                       \
          std::string,                                                \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&  \
          functions_of_time);                                         \
  template void test_det_jac_positive(                                \
      const std::vector<Block<DIM(data)>>& blocks, const double time, \
      const std::unordered_map<                                       \
          std::string,                                                \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&  \
          functions_of_time);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE

#define GET_FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                 \
  template void test_domain_construction<DIM(data), GET_FRAME(data)>(        \
      const Domain<DIM(data)>& domain,                                       \
      const std::vector<DirectionMap<DIM(data), BlockNeighbor<DIM(data)>>>&  \
          expected_block_neighbors,                                          \
      const std::vector<std::unordered_set<Direction<DIM(data)>>>&           \
          expected_external_boundaries,                                      \
      const std::vector<std::unique_ptr<domain::CoordinateMapBase<           \
          Frame::BlockLogical, GET_FRAME(data), DIM(data)>>>& expected_maps, \
      const double time,                                                     \
      const std::unordered_map<                                              \
          std::string,                                                       \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&         \
          functions_of_time,                                                 \
      const std::vector<std::unique_ptr<domain::CoordinateMapBase<           \
          Frame::Grid, Frame::Inertial, DIM(data)>>>&                        \
          expected_logical_to_grid_maps);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef INSTANTIATE
#undef GET_FRAME

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_BASIS_VECTOR(_, data)                          \
  template tnsr::i<DTYPE(data), DIM(data)> euclidean_basis_vector( \
      const Direction<DIM(data)>& direction,                       \
      const DTYPE(data) & used_for_size);                          \
  template tnsr::i<DTYPE(data), DIM(data)> unit_basis_form(        \
      const Direction<DIM(data)>& direction,                       \
      const tnsr::II<DTYPE(data), DIM(data)>& inv_spatial_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE_BASIS_VECTOR, (1, 2, 3),
                        (double, DataVector))

#undef DIM
#undef DTYPE
#undef INSTANTIATE_BASIS_VECTOR
