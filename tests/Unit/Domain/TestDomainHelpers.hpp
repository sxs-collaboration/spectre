// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines helper functions for testing Domain and DomainCreators.

#pragma once

#include "Domain/Domain.hpp"

template <size_t Dim>
void test_domain_construction(
    const Domain<Dim, Frame::Inertial>& domain,
    const std::vector<std::unordered_map<Direction<Dim>, BlockNeighbor<Dim>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<Direction<Dim>>>&
        expected_external_boundaries) {
  const auto& blocks = domain.blocks();
  CHECK(blocks.size() == expected_external_boundaries.size());
  CHECK(blocks.size() == expected_block_neighbors.size());
  for (size_t i = 0; i < blocks.size(); ++i) {
    const auto& block = blocks[i];
    CHECK(block.id() == i);
    CHECK(block.neighbors() == expected_block_neighbors[i]);
    CHECK(block.external_boundaries() == expected_external_boundaries[i]);
  }
}
