// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/Domain/DomainTestHelpers.hpp"

#include <catch.hpp>
#include <typeinfo>

#include "Domain/Block.hpp"
#include "Domain/BlockNeighbor.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Domain.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"

template <size_t VolumeDim>
void test_domain_construction(
    const Domain<VolumeDim, Frame::Inertial>& domain,
    const std::vector<
        std::unordered_map<Direction<VolumeDim>, BlockNeighbor<VolumeDim>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<Direction<VolumeDim>>>&
        expected_external_boundaries,
    const std::vector<std::unique_ptr<
        CoordinateMapBase<Frame::Logical, Frame::Inertial, VolumeDim>>>&
        expected_maps) {
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
}

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
          expected_maps);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
