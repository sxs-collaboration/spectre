// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <memory>

#include "Domain/CoordinateMaps/AffineMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Domain.hpp"
#include "Utilities/MakeVector.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
template <size_t Dim>
void test_domain_construction(
    const Domain<Dim, Frame::Grid>& domain,
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

void test_1d_domains() {
  {
    PUPable_reg(SINGLE_ARG(
        CoordinateMap<Frame::Logical, Frame::Grid, CoordinateMaps::AffineMap>));

    // Test construction of two intervals which have anti-aligned logical axes.
    const Domain<1, Frame::Grid> domain(
        make_vector<
            std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Grid, 1>>>(
            std::make_unique<CoordinateMap<Frame::Logical, Frame::Grid,
                                           CoordinateMaps::AffineMap>>(
                make_coordinate_map<Frame::Logical, Frame::Grid>(
                    CoordinateMaps::AffineMap{-1., 1., -2., 0.})),
            std::make_unique<CoordinateMap<Frame::Logical, Frame::Grid,
                                           CoordinateMaps::AffineMap>>(
                make_coordinate_map<Frame::Logical, Frame::Grid>(
                    CoordinateMaps::AffineMap{-1., 1., 0., 2.}))),
        std::vector<std::array<size_t, 2>>{{{1, 2}}, {{3, 2}}});

    const Orientation<1> unaligned_orientation{{{Direction<1>::lower_xi()}},
                                               {{Direction<1>::upper_xi()}}};

    const std::vector<std::unordered_map<Direction<1>, BlockNeighbor<1>>>
        expected_neighbors{{{Direction<1>::upper_xi(),
                             BlockNeighbor<1>{1, unaligned_orientation}}},
                           {{Direction<1>::upper_xi(),
                             BlockNeighbor<1>{0, unaligned_orientation}}}};

    const std::vector<std::unordered_set<Direction<1>>> expected_boundaries{
        {Direction<1>::lower_xi()}, {Direction<1>::lower_xi()}};

    test_domain_construction(domain, expected_neighbors, expected_boundaries);
    test_domain_construction(serialize_and_deserialize(domain),
                             expected_neighbors, expected_boundaries);

    auto vector_of_blocks = [&expected_neighbors]() {
      std::vector<Block<1, Frame::Grid>> vec;
      vec.emplace_back(Block<1, Frame::Grid>{
          std::make_unique<CoordinateMap<Frame::Logical, Frame::Grid,
                                         CoordinateMaps::AffineMap>>(
              make_coordinate_map<Frame::Logical, Frame::Grid>(
                  CoordinateMaps::AffineMap{-1., 1., -2., 0.})),
          0, expected_neighbors[0]});
      vec.emplace_back(Block<1, Frame::Grid>{
          std::make_unique<CoordinateMap<Frame::Logical, Frame::Grid,
                                         CoordinateMaps::AffineMap>>(
              make_coordinate_map<Frame::Logical, Frame::Grid>(
                  CoordinateMaps::AffineMap{-1., 1., 0., 2.})),
          1, expected_neighbors[1]});
      return vec;
    }();

    test_domain_construction(
        Domain<1, Frame::Grid>{std::move(vector_of_blocks)}, expected_neighbors,
        expected_boundaries);

    CHECK(get_output(domain) ==
          "Domain with 2 blocks:\n"
          "Block 0:\n"
          "Neighbors:\n"
          "+0: Id = 1; orientation = (-0)\n"
          "External boundaries: (-0)\n\n"
          "Block 1:\n"
          "Neighbors:\n"
          "+0: Id = 0; orientation = (-0)\n"
          "External boundaries: (-0)\n\n");
  }

  {
    // Test construction of a periodic domain
    const Domain<1, Frame::Grid> domain{
        make_vector<
            std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Grid, 1>>>(
            std::make_unique<CoordinateMap<Frame::Logical, Frame::Grid,
                                           CoordinateMaps::AffineMap>>(
                make_coordinate_map<Frame::Logical, Frame::Grid>(
                    CoordinateMaps::AffineMap{-1., 1., -2., 2.}))),
        std::vector<std::array<size_t, 2>>{{{1, 2}}},
        std::vector<std::vector<size_t>>{{1}, {2}}};

    const auto expected_neighbors = []() {
      Orientation<1> orientation{{{Direction<1>::lower_xi()}},
                                 {{Direction<1>::lower_xi()}}};
      return std::vector<std::unordered_map<Direction<1>, BlockNeighbor<1>>>{
          {{Direction<1>::lower_xi(), BlockNeighbor<1>{0, orientation}},
           {Direction<1>::upper_xi(), BlockNeighbor<1>{0, orientation}}}};
    }();

    test_domain_construction(domain, expected_neighbors,
                             std::vector<std::unordered_set<Direction<1>>>{1});
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Domain", "[Domain][Unit]") { test_1d_domains(); }
