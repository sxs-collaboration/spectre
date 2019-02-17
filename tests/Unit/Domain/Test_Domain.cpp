// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/BlockNeighbor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/StdHelpers.hpp"
#include "tests/Unit/Domain/DomainTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace domain {
namespace {
void test_1d_domains() {
  {
    CHECK(Tags::Domain<1, Frame::Logical>::name() == "Domain");
    CHECK(Tags::Domain<1, Frame::Inertial>::name() == "Domain");
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

    const std::vector<DirectionMap<1, BlockNeighbor<1>>> expected_neighbors{
        {{Direction<1>::upper_xi(),
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

    CHECK(get_output(domain) == "Domain with 2 blocks:\n" +
                                    get_output(domain.blocks()[0]) + "\n" +
                                    get_output(domain.blocks()[1]) + "\n");
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
      return std::vector<DirectionMap<1, BlockNeighbor<1>>>{
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
SPECTRE_TEST_CASE("Unit.Domain.Domain.Rectilinear1D1", "[Domain][Unit]") {
  SECTION("Aligned domain.") {
    const auto domain = rectilinear_domain<1, Frame::Inertial>(
        Index<1>{3}, std::array<std::vector<double>, 1>{{{0.0, 1.0, 2.0, 3.0}}},
        {}, {}, {{false}}, {}, true);
    const OrientationMap<1> aligned{};
    std::vector<DirectionMap<1, BlockNeighbor<1>>> expected_block_neighbors{
        {{Direction<1>::upper_xi(), {1, aligned}}},
        {{Direction<1>::lower_xi(), {0, aligned}},
         {Direction<1>::upper_xi(), {2, aligned}}},
        {{Direction<1>::lower_xi(), {1, aligned}}}};
    const std::vector<std::unordered_set<Direction<1>>>
        expected_external_boundaries{
            {{Direction<1>::lower_xi()}}, {}, {{Direction<1>::upper_xi()}}};
    for (size_t i = 0; i < domain.blocks().size(); i++) {
      INFO(i);
      CHECK(domain.blocks()[i].external_boundaries() ==
            expected_external_boundaries[i]);
    }
    for (size_t i = 0; i < domain.blocks().size(); i++) {
      INFO(i);
      CHECK(domain.blocks()[i].neighbors() == expected_block_neighbors[i]);
    }
  }
  SECTION("Antialigned domain.") {
    const OrientationMap<1> aligned{};
    const OrientationMap<1> antialigned{
        std::array<Direction<1>, 1>{{Direction<1>::lower_xi()}}};
    const auto domain = rectilinear_domain<1, Frame::Inertial>(
        Index<1>{3}, std::array<std::vector<double>, 1>{{{0.0, 1.0, 2.0, 3.0}}},
        {}, std::vector<OrientationMap<1>>{aligned, antialigned, aligned},
        {{false}}, {}, true);
    std::vector<DirectionMap<1, BlockNeighbor<1>>> expected_block_neighbors{
        {{Direction<1>::upper_xi(), {1, antialigned}}},
        {{Direction<1>::lower_xi(), {2, antialigned}},
         {Direction<1>::upper_xi(), {0, antialigned}}},
        {{Direction<1>::lower_xi(), {1, antialigned}}}};
    const std::vector<std::unordered_set<Direction<1>>>
        expected_external_boundaries{
            {{Direction<1>::lower_xi()}}, {}, {{Direction<1>::upper_xi()}}};
    for (size_t i = 0; i < domain.blocks().size(); i++) {
      INFO(i);
      CHECK(domain.blocks()[i].external_boundaries() ==
            expected_external_boundaries[i]);
    }
    for (size_t i = 0; i < domain.blocks().size(); i++) {
      INFO(i);
      CHECK(domain.blocks()[i].neighbors() == expected_block_neighbors[i]);
    }
  }
}

SPECTRE_TEST_CASE("Unit.Domain.Domain.Rectilinear2D", "[Domain][Unit]") {
  CHECK(Tags::Domain<2, Frame::Inertial>::name() == "Domain");
  const OrientationMap<2> half_turn{std::array<Direction<2>, 2>{
      {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}};
  const OrientationMap<2> quarter_turn_cw{std::array<Direction<2>, 2>{
      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}};
  const OrientationMap<2> quarter_turn_ccw{std::array<Direction<2>, 2>{
      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}};
  auto orientations_of_all_blocks =
      std::vector<OrientationMap<2>>{4, OrientationMap<2>{}};
  orientations_of_all_blocks[1] = half_turn;
  orientations_of_all_blocks[2] = quarter_turn_cw;
  orientations_of_all_blocks[3] = quarter_turn_ccw;

  const auto domain = rectilinear_domain<2, Frame::Inertial>(
      Index<2>{2, 2},
      std::array<std::vector<double>, 2>{{{0.0, 1.0, 2.0}, {0.0, 1.0, 2.0}}},
      {}, orientations_of_all_blocks);
  std::vector<DirectionMap<2, BlockNeighbor<2>>> expected_block_neighbors{
      {{Direction<2>::upper_xi(), {1, half_turn}},
       {Direction<2>::upper_eta(), {2, quarter_turn_cw}}},
      {{Direction<2>::upper_xi(), {0, half_turn}},
       {Direction<2>::lower_eta(), {3, quarter_turn_cw}}},
      {{Direction<2>::upper_xi(), {0, quarter_turn_ccw}},
       {Direction<2>::upper_eta(), {3, half_turn}}},
      {{Direction<2>::lower_xi(), {1, quarter_turn_ccw}},
       {Direction<2>::upper_eta(), {2, half_turn}}}};
  const std::vector<std::unordered_set<Direction<2>>>
      expected_external_boundaries{
          {{Direction<2>::lower_xi(), Direction<2>::lower_eta()}},
          {{Direction<2>::upper_eta(), Direction<2>::lower_xi()}},
          {{Direction<2>::lower_xi(), Direction<2>::lower_eta()}},
          {{Direction<2>::upper_xi(), Direction<2>::lower_eta()}}};
  for (size_t i = 0; i < domain.blocks().size(); i++) {
    INFO(i);
    CHECK(domain.blocks()[i].external_boundaries() ==
          expected_external_boundaries[i]);
  }
  for (size_t i = 0; i < domain.blocks().size(); i++) {
    INFO(i);
    CHECK(domain.blocks()[i].neighbors() == expected_block_neighbors[i]);
  }
}

SPECTRE_TEST_CASE("Unit.Domain.Domain.Rectilinear3D", "[Domain][Unit]") {
  CHECK(Tags::Domain<3, Frame::Inertial>::name() == "Domain");
  const OrientationMap<3> aligned{};
  const OrientationMap<3> quarter_turn_cw_xi{std::array<Direction<3>, 3>{
      {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
       Direction<3>::lower_eta()}}};
  auto orientations_of_all_blocks =
      std::vector<OrientationMap<3>>{aligned, quarter_turn_cw_xi};

  const auto domain = rectilinear_domain<3, Frame::Inertial>(
      Index<3>{2, 1, 1},
      std::array<std::vector<double>, 3>{
          {{0.0, 1.0, 2.0}, {0.0, 1.0}, {0.0, 1.0}}},
      {}, orientations_of_all_blocks);
  std::vector<DirectionMap<3, BlockNeighbor<3>>> expected_block_neighbors{
      {{Direction<3>::upper_xi(), {1, quarter_turn_cw_xi}}},
      {{Direction<3>::lower_xi(), {0, quarter_turn_cw_xi.inverse_map()}}}};
  const std::vector<std::unordered_set<Direction<3>>>
      expected_external_boundaries{
          {{Direction<3>::lower_xi(), Direction<3>::lower_eta(),
            Direction<3>::upper_eta(), Direction<3>::lower_zeta(),
            Direction<3>::upper_zeta()}},
          {{Direction<3>::upper_xi(), Direction<3>::lower_eta(),
            Direction<3>::upper_eta(), Direction<3>::lower_zeta(),
            Direction<3>::upper_zeta()}}};
  for (size_t i = 0; i < domain.blocks().size(); i++) {
    INFO(i);
    CHECK(domain.blocks()[i].external_boundaries() ==
          expected_external_boundaries[i]);
  }
  for (size_t i = 0; i < domain.blocks().size(); i++) {
    INFO(i);
    CHECK(domain.blocks()[i].neighbors() == expected_block_neighbors[i]);
  }
}

// [[OutputRegex, Must pass same number of maps as block corner sets]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.Domain.BadArgs", "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  // NOLINTNEXTLINE(misc-unused-raii)
  Domain<1, Frame::Inertial>(
      make_vector(make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
                      CoordinateMaps::Affine{-1., 1., -1., 1.}),
                  make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
                      CoordinateMaps::Affine{-1., 1., -1., 1.})),
      std::vector<std::array<size_t, 2>>{{{1, 2}}});
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
}  // namespace domain
