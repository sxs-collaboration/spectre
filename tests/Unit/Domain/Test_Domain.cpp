// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>
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
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/OrientationMap.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeVector.hpp"
#include "tests/Unit/Domain/DomainTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
void test_1d_domains() {
  {
    PUPable_reg(
        SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                         domain::CoordinateMaps::Affine>));

    // Test construction of two intervals which have anti-aligned logical axes.
    const domain::Domain<1, Frame::Inertial> domain(
        make_vector<std::unique_ptr<
            domain::CoordinateMapBase<Frame::Logical, Frame::Inertial, 1>>>(
            std::make_unique<
                domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                      domain::CoordinateMaps::Affine>>(
                domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
                    domain::CoordinateMaps::Affine{-1., 1., -2., 0.})),
            std::make_unique<
                domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                      domain::CoordinateMaps::Affine>>(
                domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
                    domain::CoordinateMaps::Affine{-1., 1., 0., 2.}))),
        std::vector<std::array<size_t, 2>>{{{1, 2}}, {{3, 2}}});

    const domain::OrientationMap<1> unaligned_orientation{
        {{domain::Direction<1>::lower_xi()}},
        {{domain::Direction<1>::upper_xi()}}};

    const std::vector<
        std::unordered_map<domain::Direction<1>, domain::BlockNeighbor<1>>>
        expected_neighbors{
            {{domain::Direction<1>::upper_xi(),
              domain::BlockNeighbor<1>{1, unaligned_orientation}}},
            {{domain::Direction<1>::upper_xi(),
              domain::BlockNeighbor<1>{0, unaligned_orientation}}}};

    const std::vector<std::unordered_set<domain::Direction<1>>>
        expected_boundaries{{domain::Direction<1>::lower_xi()},
                            {domain::Direction<1>::lower_xi()}};

    const auto expected_maps = make_vector(
        domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            domain::CoordinateMaps::Affine{-1., 1., -2., 0.}),
        domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            domain::CoordinateMaps::Affine{-1., 1., 0., 2.}));

    test_domain_construction(domain, expected_neighbors, expected_boundaries,
                             expected_maps);

    test_domain_construction(serialize_and_deserialize(domain),
                             expected_neighbors, expected_boundaries,
                             expected_maps);

    auto vector_of_blocks = [&expected_neighbors]() {
      std::vector<domain::Block<1, Frame::Inertial>> vec;
      vec.emplace_back(domain::Block<1, Frame::Inertial>{
          std::make_unique<domain::CoordinateMap<
              Frame::Logical, Frame::Inertial, domain::CoordinateMaps::Affine>>(
              domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
                  domain::CoordinateMaps::Affine{-1., 1., -2., 0.})),
          0, expected_neighbors[0]});
      vec.emplace_back(domain::Block<1, Frame::Inertial>{
          std::make_unique<domain::CoordinateMap<
              Frame::Logical, Frame::Inertial, domain::CoordinateMaps::Affine>>(
              domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
                  domain::CoordinateMaps::Affine{-1., 1., 0., 2.})),
          1, expected_neighbors[1]});
      return vec;
    }();

    test_domain_construction(
        domain::Domain<1, Frame::Inertial>{std::move(vector_of_blocks)},
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
    const auto expected_maps = make_vector(
        domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            domain::CoordinateMaps::Affine{-1., 1., -2., 2.}));

    const domain::Domain<1, Frame::Inertial> domain{
        make_vector<std::unique_ptr<
            domain::CoordinateMapBase<Frame::Logical, Frame::Inertial, 1>>>(
            std::make_unique<
                domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                      domain::CoordinateMaps::Affine>>(
                domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
                    domain::CoordinateMaps::Affine{-1., 1., -2., 2.}))),
        std::vector<std::array<size_t, 2>>{{{1, 2}}},
        std::vector<domain::PairOfFaces>{{{1}, {2}}}};

    const auto expected_neighbors = []() {
      domain::OrientationMap<1> orientation{
          {{domain::Direction<1>::lower_xi()}},
          {{domain::Direction<1>::lower_xi()}}};
      return std::vector<
          std::unordered_map<domain::Direction<1>, domain::BlockNeighbor<1>>>{
          {{domain::Direction<1>::lower_xi(),
            domain::BlockNeighbor<1>{0, orientation}},
           {domain::Direction<1>::upper_xi(),
            domain::BlockNeighbor<1>{0, orientation}}}};
    }();

    test_domain_construction(
        domain, expected_neighbors,
        std::vector<std::unordered_set<domain::Direction<1>>>{1},
        expected_maps);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Domain", "[Domain][Unit]") { test_1d_domains(); }
SPECTRE_TEST_CASE("Unit.Domain.Domain.Rectilinear1D1", "[Domain][Unit]") {
  SECTION("Aligned domain.") {
    const auto domain = domain::rectilinear_domain<1, Frame::Inertial>(
        Index<1>{3}, std::array<std::vector<double>, 1>{{{0.0, 1.0, 2.0, 3.0}}},
        {}, {}, {{false}}, {}, true);
    const domain::OrientationMap<1> aligned{};
    std::vector<
        std::unordered_map<domain::Direction<1>, domain::BlockNeighbor<1>>>
        expected_block_neighbors{
            {{domain::Direction<1>::upper_xi(), {1, aligned}}},
            {{domain::Direction<1>::lower_xi(), {0, aligned}},
             {domain::Direction<1>::upper_xi(), {2, aligned}}},
            {{domain::Direction<1>::lower_xi(), {1, aligned}}}};
    const std::vector<std::unordered_set<domain::Direction<1>>>
        expected_external_boundaries{{{domain::Direction<1>::lower_xi()}},
                                     {},
                                     {{domain::Direction<1>::upper_xi()}}};
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
    const domain::OrientationMap<1> aligned{};
    const domain::OrientationMap<1> antialigned{
        std::array<domain::Direction<1>, 1>{
            {domain::Direction<1>::lower_xi()}}};
    const auto domain = domain::rectilinear_domain<1, Frame::Inertial>(
        Index<1>{3}, std::array<std::vector<double>, 1>{{{0.0, 1.0, 2.0, 3.0}}},
        {},
        std::vector<domain::OrientationMap<1>>{aligned, antialigned, aligned},
        {{false}}, {}, true);
    std::vector<
        std::unordered_map<domain::Direction<1>, domain::BlockNeighbor<1>>>
        expected_block_neighbors{
            {{domain::Direction<1>::upper_xi(), {1, antialigned}}},
            {{domain::Direction<1>::lower_xi(), {2, antialigned}},
             {domain::Direction<1>::upper_xi(), {0, antialigned}}},
            {{domain::Direction<1>::lower_xi(), {1, antialigned}}}};
    const std::vector<std::unordered_set<domain::Direction<1>>>
        expected_external_boundaries{{{domain::Direction<1>::lower_xi()}},
                                     {},
                                     {{domain::Direction<1>::upper_xi()}}};
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
  const domain::OrientationMap<2> half_turn{std::array<domain::Direction<2>, 2>{
      {domain::Direction<2>::lower_xi(), domain::Direction<2>::lower_eta()}}};
  const domain::OrientationMap<2> quarter_turn_cw{
      std::array<domain::Direction<2>, 2>{{domain::Direction<2>::upper_eta(),
                                           domain::Direction<2>::lower_xi()}}};
  const domain::OrientationMap<2> quarter_turn_ccw{
      std::array<domain::Direction<2>, 2>{{domain::Direction<2>::lower_eta(),
                                           domain::Direction<2>::upper_xi()}}};
  auto orientations_of_all_blocks =
      std::vector<domain::OrientationMap<2>>{4, domain::OrientationMap<2>{}};
  orientations_of_all_blocks[1] = half_turn;
  orientations_of_all_blocks[2] = quarter_turn_cw;
  orientations_of_all_blocks[3] = quarter_turn_ccw;

  const auto domain = domain::rectilinear_domain<2, Frame::Inertial>(
      Index<2>{2, 2},
      std::array<std::vector<double>, 2>{{{0.0, 1.0, 2.0}, {0.0, 1.0, 2.0}}},
      {}, orientations_of_all_blocks);
  std::vector<
      std::unordered_map<domain::Direction<2>, domain::BlockNeighbor<2>>>
      expected_block_neighbors{
          {{domain::Direction<2>::upper_xi(), {1, half_turn}},
           {domain::Direction<2>::upper_eta(), {2, quarter_turn_cw}}},
          {{domain::Direction<2>::upper_xi(), {0, half_turn}},
           {domain::Direction<2>::lower_eta(), {3, quarter_turn_cw}}},
          {{domain::Direction<2>::upper_xi(), {0, quarter_turn_ccw}},
           {domain::Direction<2>::upper_eta(), {3, half_turn}}},
          {{domain::Direction<2>::lower_xi(), {1, quarter_turn_ccw}},
           {domain::Direction<2>::upper_eta(), {2, half_turn}}}};
  const std::vector<std::unordered_set<domain::Direction<2>>>
      expected_external_boundaries{{{domain::Direction<2>::lower_xi(),
                                     domain::Direction<2>::lower_eta()}},
                                   {{domain::Direction<2>::upper_eta(),
                                     domain::Direction<2>::lower_xi()}},
                                   {{domain::Direction<2>::lower_xi(),
                                     domain::Direction<2>::lower_eta()}},
                                   {{domain::Direction<2>::upper_xi(),
                                     domain::Direction<2>::lower_eta()}}};
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
  const domain::OrientationMap<3> aligned{};
  const domain::OrientationMap<3> quarter_turn_cw_xi{
      std::array<domain::Direction<3>, 3>{{domain::Direction<3>::upper_xi(),
                                           domain::Direction<3>::upper_zeta(),
                                           domain::Direction<3>::lower_eta()}}};
  auto orientations_of_all_blocks =
      std::vector<domain::OrientationMap<3>>{aligned, quarter_turn_cw_xi};

  const auto domain = domain::rectilinear_domain<3, Frame::Inertial>(
      Index<3>{2, 1, 1},
      std::array<std::vector<double>, 3>{
          {{0.0, 1.0, 2.0}, {0.0, 1.0}, {0.0, 1.0}}},
      {}, orientations_of_all_blocks);
  std::vector<
      std::unordered_map<domain::Direction<3>, domain::BlockNeighbor<3>>>
      expected_block_neighbors{
          {{domain::Direction<3>::upper_xi(), {1, quarter_turn_cw_xi}}},
          {{domain::Direction<3>::lower_xi(),
            {0, quarter_turn_cw_xi.inverse_map()}}}};
  const std::vector<std::unordered_set<domain::Direction<3>>>
      expected_external_boundaries{
          {{domain::Direction<3>::lower_xi(), domain::Direction<3>::lower_eta(),
            domain::Direction<3>::upper_eta(),
            domain::Direction<3>::lower_zeta(),
            domain::Direction<3>::upper_zeta()}},
          {{domain::Direction<3>::upper_xi(), domain::Direction<3>::lower_eta(),
            domain::Direction<3>::upper_eta(),
            domain::Direction<3>::lower_zeta(),
            domain::Direction<3>::upper_zeta()}}};
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
  domain::Domain<1, Frame::Inertial>(
      make_vector(
          domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
              domain::CoordinateMaps::Affine{-1., 1., -1., 1.}),
          domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
              domain::CoordinateMaps::Affine{-1., 1., -1., 1.})),
      std::vector<std::array<size_t, 2>>{{{1, 2}}});
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
