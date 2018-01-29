// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <memory>

#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Domain.hpp"
#include "Utilities/MakeVector.hpp"
#include "tests/Unit/Domain/DomainTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
void test_1d_domains() {
  {
    PUPable_reg(SINGLE_ARG(CoordinateMap<Frame::Logical, Frame::Inertial,
                                         CoordinateMaps::Affine>));

    // Test construction of two intervals which have anti-aligned logical axes.
    const Domain<1, Frame::Inertial> domain(
        make_vector<std::unique_ptr<
            CoordinateMapBase<Frame::Logical, Frame::Inertial, 1>>>(
            std::make_unique<CoordinateMap<Frame::Logical, Frame::Inertial,
                                           CoordinateMaps::Affine>>(
                make_coordinate_map<Frame::Logical, Frame::Inertial>(
                    CoordinateMaps::Affine{-1., 1., -2., 0.})),
            std::make_unique<CoordinateMap<Frame::Logical, Frame::Inertial,
                                           CoordinateMaps::Affine>>(
                make_coordinate_map<Frame::Logical, Frame::Inertial>(
                    CoordinateMaps::Affine{-1., 1., 0., 2.}))),
        std::vector<std::array<size_t, 2>>{{{1, 2}}, {{3, 2}}});

    const OrientationMap<1> unaligned_orientation{{{Direction<1>::lower_xi()}},
                                                  {{Direction<1>::upper_xi()}}};

    const std::vector<std::unordered_map<Direction<1>, BlockNeighbor<1>>>
        expected_neighbors{{{Direction<1>::upper_xi(),
                             BlockNeighbor<1>{1, unaligned_orientation}}},
                           {{Direction<1>::upper_xi(),
                             BlockNeighbor<1>{0, unaligned_orientation}}}};

    const std::vector<std::unordered_set<Direction<1>>> expected_boundaries{
        {Direction<1>::lower_xi()}, {Direction<1>::lower_xi()}};

    const auto expected_maps =
        make_vector(make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
                        CoordinateMaps::Affine{-1., 1., -2., 0.}),
                    make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
                        CoordinateMaps::Affine{-1., 1., 0., 2.}));

    test_domain_construction(domain, expected_neighbors, expected_boundaries,
                             expected_maps);

    test_domain_construction(serialize_and_deserialize(domain),
                             expected_neighbors, expected_boundaries,
                             expected_maps);

    auto vector_of_blocks = [&expected_neighbors]() {
      std::vector<Block<1, Frame::Inertial>> vec;
      vec.emplace_back(Block<1, Frame::Inertial>{
          std::make_unique<CoordinateMap<Frame::Logical, Frame::Inertial,
                                         CoordinateMaps::Affine>>(
              make_coordinate_map<Frame::Logical, Frame::Inertial>(
                  CoordinateMaps::Affine{-1., 1., -2., 0.})),
          0, expected_neighbors[0]});
      vec.emplace_back(Block<1, Frame::Inertial>{
          std::make_unique<CoordinateMap<Frame::Logical, Frame::Inertial,
                                         CoordinateMaps::Affine>>(
              make_coordinate_map<Frame::Logical, Frame::Inertial>(
                  CoordinateMaps::Affine{-1., 1., 0., 2.})),
          1, expected_neighbors[1]});
      return vec;
    }();

    test_domain_construction(
        Domain<1, Frame::Inertial>{std::move(vector_of_blocks)},
        expected_neighbors, expected_boundaries, expected_maps);

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
    const auto expected_maps =
        make_vector(make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            CoordinateMaps::Affine{-1., 1., -2., 2.}));

    const Domain<1, Frame::Inertial> domain{
        make_vector<std::unique_ptr<
            CoordinateMapBase<Frame::Logical, Frame::Inertial, 1>>>(
            std::make_unique<CoordinateMap<Frame::Logical, Frame::Inertial,
                                           CoordinateMaps::Affine>>(
                make_coordinate_map<Frame::Logical, Frame::Inertial>(
                    CoordinateMaps::Affine{-1., 1., -2., 2.}))),
        std::vector<std::array<size_t, 2>>{{{1, 2}}},
        std::vector<PairOfFaces>{{{1}, {2}}}};

    const auto expected_neighbors = []() {
      OrientationMap<1> orientation{{{Direction<1>::lower_xi()}},
                                    {{Direction<1>::lower_xi()}}};
      return std::vector<std::unordered_map<Direction<1>, BlockNeighbor<1>>>{
          {{Direction<1>::lower_xi(), BlockNeighbor<1>{0, orientation}},
           {Direction<1>::upper_xi(), BlockNeighbor<1>{0, orientation}}}};
    }();

    test_domain_construction(domain, expected_neighbors,
                             std::vector<std::unordered_set<Direction<1>>>{1},
                             expected_maps);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Domain", "[Domain][Unit]") { test_1d_domains(); }
