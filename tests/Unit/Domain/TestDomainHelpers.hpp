// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines helper functions for testing Domain and DomainCreators.

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BlockNeighbor.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Domain.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"

template <size_t Dim, typename Map>
void test_domain_construction(
    const Domain<Dim, Frame::Inertial>& domain,
    const std::vector<std::unordered_map<Direction<Dim>, BlockNeighbor<Dim>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<Direction<Dim>>>&
        expected_external_boundaries,
    const std::vector<Map>& expected_maps) {
  const auto& blocks = domain.blocks();
  CHECK(blocks.size() == expected_external_boundaries.size());
  CHECK(blocks.size() == expected_block_neighbors.size());
  CHECK(blocks.size() == expected_maps.size());
  for (size_t i = 0; i < blocks.size(); ++i) {
    const auto& block = blocks[i];
    CHECK(block.id() == i);
    CHECK(block.neighbors() == expected_block_neighbors[i]);
    CHECK(block.external_boundaries() == expected_external_boundaries[i]);
    CHECK(are_maps_equal(expected_maps[i], block.coordinate_map()));
  }
}
