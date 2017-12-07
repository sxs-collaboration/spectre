// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/Domain/DomainTestHelpers.hpp"

#include <catch.hpp>
#include <typeinfo>

#include "Domain/Block.hpp"
#include "Domain/BlockNeighbor.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainCreators/RegisterDerivedWithCharm.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Domain/SegmentId.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"

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
  return boost::rational<size_t>(1, two_to_the(sum_of_refinement_levels));
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

template <size_t VolumeDim>
void test_domain_connectivity(
    const Domain<VolumeDim, Frame::Inertial>& domain,
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
      const auto& direction = direction_neighbors.first;
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
}  // namespace

template <size_t VolumeDim>
void test_domain_construction(
    const Domain<VolumeDim, Frame::Inertial>& domain,
    const std::vector<
        std::unordered_map<Direction<VolumeDim>, BlockNeighbor<VolumeDim>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<Direction<VolumeDim>>>&
        expected_external_boundaries,
    const std::vector<std::unique_ptr<CoordinateMapBase<
        Frame::Logical, Frame::Inertial, VolumeDim>>>& expected_maps) noexcept {
  const auto& blocks = domain.blocks();
  CHECK(blocks.size() == expected_external_boundaries.size());
  CHECK(blocks.size() == expected_block_neighbors.size());
  CHECK(blocks.size() == expected_maps.size());
  for (size_t i = 0; i < blocks.size(); ++i) {
    const auto& block = blocks[i];
    CHECK(block.id() == i);
    CHECK(block.neighbors() == expected_block_neighbors[i]);
    CHECK(block.external_boundaries() == expected_external_boundaries[i]);
    CHECK(typeid(*expected_maps[i]) == typeid(block.coordinate_map()));
    check_if_maps_are_equal(*expected_maps[i], block.coordinate_map());
  }
  DomainCreators::register_derived_with_charm();
  test_serialization(domain);
  // test operator !=
  CHECK_FALSE(domain != domain);
}

template <size_t VolumeDim>
boost::rational<size_t> fraction_of_block_volume(
    const ElementId<VolumeDim>& element_id) noexcept {
  const auto& segment_ids = element_id.segment_ids();
  size_t sum_of_refinement_levels = 0;
  for (const auto& segment_id : segment_ids) {
    sum_of_refinement_levels += segment_id.refinement_level();
  }
  return boost::rational<size_t>(1, two_to_the(sum_of_refinement_levels));
}

template <size_t VolumeDim>
void test_initial_domain(const Domain<VolumeDim, Frame::Inertial>& domain,
                         const std::vector<std::array<size_t, VolumeDim>>&
                             initial_refinement_levels) noexcept {
  const auto element_ids = initial_element_ids(initial_refinement_levels);
  const auto& blocks = domain.blocks();
  std::unordered_map<ElementId<VolumeDim>, Element<VolumeDim>> elements;
  for (const auto& element_id : element_ids) {
    elements.emplace(
        element_id,
        create_initial_element(element_id, blocks[element_id.block_id()]));
  }
  test_domain_connectivity(domain, elements);
  test_refinement_levels_of_neighbors<0>(elements);
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void test_domain_construction<DIM(data)>(                           \
      const Domain<DIM(data), Frame::Inertial>& domain,                        \
      const std::vector<                                                       \
          std::unordered_map<Direction<DIM(data)>, BlockNeighbor<DIM(data)>>>& \
          expected_block_neighbors,                                            \
      const std::vector<std::unordered_set<Direction<DIM(data)>>>&             \
          expected_external_boundaries,                                        \
      const std::vector<std::unique_ptr<                                       \
          CoordinateMapBase<Frame::Logical, Frame::Inertial, DIM(data)>>>&     \
          expected_maps) noexcept;                                             \
  template boost::rational<size_t> fraction_of_block_volume(                   \
      const ElementId<DIM(data)>& element_id) noexcept;                        \
  template void test_initial_domain(                                           \
      const Domain<DIM(data), Frame::Inertial>& domain,                        \
      const std::vector<std::array<size_t, DIM(data)>>&                        \
          initial_refinement_levels) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond
