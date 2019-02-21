// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "tests/Unit/Domain/DomainTestHelpers.hpp"

#include <algorithm>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"     // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Domain/Block.hpp"                  // IWYU pragma: keep
#include "Domain/BlockNeighbor.hpp"          // IWYU pragma: keep
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"        // IWYU pragma: keep
#include "Domain/DomainHelpers.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Domain/Neighbors.hpp"  // IWYU pragma: keep
#include "Domain/OrientationMap.hpp"
#include "Domain/Side.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

template <size_t VolumeDim>
class Element;

namespace domain {
namespace {
template <size_t VolumeDim>
boost::rational<size_t> fraction_of_block_face_area(
    const ElementId<VolumeDim>& element_id,
    const Direction<VolumeDim>& direction) noexcept {
  const auto& segment_ids = element_id.segment_ids();
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
    const Direction<VolumeDim>& direction_to_me_in_neighbor) noexcept {
  size_t number_of_matches = 0;
  for (const auto& id_of_neighbor_of_neighbor :
       my_neighbor.neighbors().at(direction_to_me_in_neighbor).ids()) {
    if (my_id == id_of_neighbor_of_neighbor) {
      ++number_of_matches;
    }
  }
  return 1 == number_of_matches;
}

template <size_t VolumeDim, typename Fr>
void test_domain_connectivity(
    const Domain<VolumeDim, Fr>& domain,
    const std::unordered_map<ElementId<VolumeDim>, Element<VolumeDim>>&
        elements_in_domain) noexcept {
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
SPECTRE_ALWAYS_INLINE void check_if_levels_are_within(
    const size_t level_1, const size_t level_2) noexcept {
  CHECK(level_1 <= level_2 + AllowedDifference);
  CHECK(level_2 <= level_1 + AllowedDifference);
}

template <>
SPECTRE_ALWAYS_INLINE void check_if_levels_are_within<0>(
    const size_t level_1, const size_t level_2) noexcept {
  CHECK(level_1 == level_2);
}

template <size_t AllowedDifference, size_t VolumeDim>
void test_refinement_levels_of_neighbors(
    const std::unordered_map<ElementId<VolumeDim>, Element<VolumeDim>>&
        elements) noexcept {
  for (const auto& key_value : elements) {
    const auto& element_id = key_value.first;
    const auto& element = key_value.second;
    for (const auto& direction_neighbors : element.neighbors()) {
      const auto& neighbors = direction_neighbors.second;
      const auto& orientation = neighbors.orientation();
      for (size_t d = 0; d < VolumeDim; ++d) {
        const size_t my_dim_in_neighbor = orientation(d);
        const size_t my_level =
            gsl::at(element_id.segment_ids(), d).refinement_level();
        for (const auto neighbor_id : neighbors.ids()) {
          const size_t neighbor_level =
              gsl::at(neighbor_id.segment_ids(), my_dim_in_neighbor)
                  .refinement_level();
          check_if_levels_are_within<AllowedDifference>(my_level,
                                                        neighbor_level);
        }
      }
    }
  }
}

template <size_t VolumeDim, typename TargetFrame>
bool blocks_are_neighbors(
    const Block<VolumeDim, TargetFrame>& host_block,
    const Block<VolumeDim, TargetFrame>& neighbor_block) noexcept {
  for (const auto& neighbor : host_block.neighbors()) {
    if (neighbor.second.id() == neighbor_block.id()) {
      return true;
    }
  }
  return false;
}

// Finds the OrientationMap of a neighboring Block relative to a host Block.
template <size_t VolumeDim, typename TargetFrame>
OrientationMap<VolumeDim> find_neighbor_orientation(
    const Block<VolumeDim, TargetFrame>& host_block,
    const Block<VolumeDim, TargetFrame>& neighbor_block) noexcept {
  for (const auto& neighbor : host_block.neighbors()) {
    if (neighbor.second.id() == neighbor_block.id()) {
      return neighbor.second.orientation();
    }
  }
  ERROR("The Block `neighbor_block` is not a neighbor of `host_block`.");
}

// Finds the Direction to the neighboring Block relative to a host Block.
template <size_t VolumeDim, typename TargetFrame>
Direction<VolumeDim> find_direction_to_neighbor(
    const Block<VolumeDim, TargetFrame>& host_block,
    const Block<VolumeDim, TargetFrame>& neighbor_block) noexcept {
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
    const tnsr::I<double, VolumeDim, Frame::Logical>& point) noexcept {
  std::array<Direction<VolumeDim>, VolumeDim> result;
  for (size_t i = 0; i < VolumeDim; i++) {
    gsl::at(result, i) =
        Direction<VolumeDim>(i, point[i] >= 0 ? Side::Upper : Side::Lower);
  }
  return result;
}

// Convert Directions to Point, for use with CoordinateMap
template <size_t VolumeDim>
tnsr::I<double, VolumeDim, Frame::Logical> get_corner_of_orthant(
    const std::array<Direction<VolumeDim>, VolumeDim>& directions) noexcept {
  tnsr::I<double, VolumeDim, Frame::Logical> result{};
  for (size_t i = 0; i < VolumeDim; i++) {
    result[gsl::at(directions, i).dimension()] =
        gsl::at(directions, i).side() == Side::Upper ? 1.0 : -1.0;
  }
  return result;
}

// The relative OrientationMap between Blocks induces a map that takes
// Points in the host Block to Points in the neighbor Block.
template <size_t VolumeDim>
tnsr::I<double, VolumeDim, Frame::Logical> point_in_neighbor_frame(
    const OrientationMap<VolumeDim>& orientation,
    const tnsr::I<double, VolumeDim, Frame::Logical>& point) noexcept {
  auto point_get_orthant = get_orthant(point);
  std::for_each(
      point_get_orthant.begin(), point_get_orthant.end(),
      [&orientation](auto& direction) { direction = orientation(direction); });
  return get_corner_of_orthant(point_get_orthant);
}

// Given two Blocks which are neighbors, computes the max separation between
// the abutting faces of the Blocks in the TargetFrame using the CoordinateMaps
// of each block.
template <size_t VolumeDim, typename TargetFrame>
double physical_separation(
    const Block<VolumeDim, TargetFrame>& block1,
    const Block<VolumeDim, TargetFrame>& block2) noexcept {
  double max_separation = 0;
  const auto direction = find_direction_to_neighbor(block1, block2);
  const auto orientation = find_neighbor_orientation(block1, block2);
  std::array<tnsr::I<double, VolumeDim, Frame::Logical>,
             two_to_the(VolumeDim - 1)>
      shared_points1{};
  std::array<tnsr::I<double, VolumeDim, Frame::Logical>,
             two_to_the(VolumeDim - 1)>
      shared_points2{};
  for (FaceCornerIterator<VolumeDim> fci(direction); fci; ++fci) {
    gsl::at(shared_points1, fci.face_index()) = fci();
  }
  for (FaceCornerIterator<VolumeDim> fci(direction.opposite()); fci; ++fci) {
    gsl::at(shared_points2, fci.face_index()) =
        point_in_neighbor_frame(orientation, fci());
  }
  const auto& map1 = block1.coordinate_map();
  const auto& map2 = block2.coordinate_map();
  for (size_t i = 0; i < two_to_the(VolumeDim - 1); i++) {
    for (size_t j = 0; j < VolumeDim; j++) {
      max_separation = std::max(
          max_separation, std::abs(map1(gsl::at(shared_points1, i)).get(j) -
                                   map2(gsl::at(shared_points2, i)).get(j)));
    }
  }
  return max_separation;
}
}  // namespace
}  // namespace domain

template <size_t VolumeDim, typename Fr>
void test_domain_construction(
    const Domain<VolumeDim, Fr>& domain,
    const std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<Direction<VolumeDim>>>&
        expected_external_boundaries,
    const std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Logical, Fr, VolumeDim>>>&
        expected_maps) noexcept {
  const auto& blocks = domain.blocks();
  CHECK(blocks.size() == expected_external_boundaries.size());
  CHECK(blocks.size() == expected_block_neighbors.size());
  CHECK(blocks.size() == expected_maps.size());
  for (size_t i = 0; i < blocks.size(); ++i) {
    const auto& block = blocks[i];
    CHECK(block.id() == i);
    CHECK(block.neighbors() == expected_block_neighbors[i]);
    CHECK(block.external_boundaries() == expected_external_boundaries[i]);
    check_if_maps_are_equal(*expected_maps[i], block.coordinate_map());
  }
  domain::creators::register_derived_with_charm();
  test_serialization(domain);
  // test operator !=
  CHECK_FALSE(domain != domain);
}

template <size_t VolumeDim, typename TargetFrame>
void test_physical_separation(
    const std::vector<Block<VolumeDim, TargetFrame>>& blocks) noexcept {
  double tolerance = 1e-10;
  for (size_t i = 0; i < blocks.size() - 1; i++) {
    for (size_t j = i + 1; j < blocks.size(); j++) {
      if (domain::blocks_are_neighbors(blocks[i], blocks[j])) {
        CHECK(domain::physical_separation(blocks[i], blocks[j]) < tolerance);
      }
    }
  }
}

template <size_t VolumeDim>
boost::rational<size_t> fraction_of_block_volume(
    const ElementId<VolumeDim>& element_id) noexcept {
  const auto& segment_ids = element_id.segment_ids();
  size_t sum_of_refinement_levels = 0;
  for (const auto& segment_id : segment_ids) {
    sum_of_refinement_levels += segment_id.refinement_level();
  }
  return {1, two_to_the(sum_of_refinement_levels)};
}

template <size_t VolumeDim, typename Frame>
void test_initial_domain(const Domain<VolumeDim, Frame>& domain,
                         const std::vector<std::array<size_t, VolumeDim>>&
                             initial_refinement_levels) noexcept {
  const auto element_ids = initial_element_ids(initial_refinement_levels);
  const auto& blocks = domain.blocks();
  CHECK(blocks.size() == initial_refinement_levels.size());
  std::unordered_map<ElementId<VolumeDim>, Element<VolumeDim>> elements;
  for (const auto& element_id : element_ids) {
    elements.emplace(
        element_id,
        create_initial_element(element_id, blocks[element_id.block_id()]));
  }
  domain::test_domain_connectivity(domain, elements);
  domain::test_refinement_levels_of_neighbors<0>(elements);
}

template <size_t SpatialDim, typename Fr>
tnsr::i<DataVector, SpatialDim, Fr> euclidean_basis_vector(
    const Direction<SpatialDim>& direction,
    const DataVector& used_for_size) noexcept {
  auto basis_vector =
      make_with_value<tnsr::i<DataVector, SpatialDim, Fr>>(used_for_size, 0.0);

  basis_vector.get(direction.axis()) =
      make_with_value<DataVector>(used_for_size, direction.sign());

  return basis_vector;
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE1(_, data)                                \
  template boost::rational<size_t> fraction_of_block_volume( \
      const ElementId<DIM(data)>& element_id) noexcept;

#define INSTANTIATE2(_, data)                                                  \
  template void test_domain_construction<DIM(data)>(                           \
      const Domain<DIM(data), FRAME(data)>& domain,                            \
      const std::vector<DirectionMap<DIM(data), BlockNeighbor<DIM(data)>>>&    \
          expected_block_neighbors,                                            \
      const std::vector<std::unordered_set<Direction<DIM(data)>>>&             \
          expected_external_boundaries,                                        \
      const std::vector<std::unique_ptr<                                       \
          domain::CoordinateMapBase<Frame::Logical, FRAME(data), DIM(data)>>>& \
          expected_maps) noexcept;                                             \
  template void test_initial_domain(                                           \
      const Domain<DIM(data), FRAME(data)>& domain,                            \
      const std::vector<std::array<size_t, DIM(data)>>&                        \
          initial_refinement_levels) noexcept;                                 \
  template void test_physical_separation(                                      \
      const std::vector<Block<DIM(data), FRAME(data)>>& blocks) noexcept;      \
  template tnsr::i<DataVector, DIM(data), FRAME(data)> euclidean_basis_vector( \
      const Direction<DIM(data)>& direction,                                   \
      const DataVector& used_for_size) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE1, (1, 2, 3))
GENERATE_INSTANTIATIONS(INSTANTIATE2, (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef DIM
#undef FRAME
#undef INSTANTIATE1
#undef INSTANTIATE2
/// \endcond
