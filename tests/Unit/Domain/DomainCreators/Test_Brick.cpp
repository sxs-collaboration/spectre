// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "Domain/CoordinateMaps/AffineMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainCreators/Brick.hpp"
#include "Utilities/MakeVector.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "tests/Unit/Domain/DomainTestHelpers.hpp"
#include "tests/Unit/TestFactoryCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
using AffineMap = CoordinateMaps::AffineMap;
using AffineMap3D =
    CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
void test_brick_construction(
    const DomainCreators::Brick& brick,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound,
    const std::array<size_t, 3>& expected_extents,
    const std::array<size_t, 3>& expected_refinement_level,
    const std::vector<std::unordered_map<Direction<3>, BlockNeighbor<3>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<Direction<3>>>&
        expected_external_boundaries) {
  const auto domain = brick.create_domain();
  const auto& block = domain.blocks()[0];
  const auto& neighbors = block.neighbors();
  const auto& external_boundaries = block.external_boundaries();

  CHECK(block.id() == 0);
  CHECK(brick.initial_extents(0) == expected_extents);
  CHECK(brick.initial_refinement_levels(0) == expected_refinement_level);

  test_domain_construction(
      domain, expected_block_neighbors, expected_external_boundaries,
      make_vector(make_coordinate_map<Frame::Logical, Frame::Inertial>(
          AffineMap3D{AffineMap{-1., 1., lower_bound[0], upper_bound[0]},
                      AffineMap{-1., 1., lower_bound[1], upper_bound[1]},
                      AffineMap{-1., 1., lower_bound[2], upper_bound[2]}})));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.Brick", "[Domain][Unit]") {
  const std::array<size_t, 3> grid_points{{4, 6, 3}}, num_elements{{3, 2, 4}};
  const std::array<double, 3> lower_bound{{-1.2, 3.0, 2.5}},
      upper_bound{{0.8, 5.0, 3.0}};
  // Default Orientation is aligned.
  const Orientation<3> aligned_orientation{};

  const DomainCreators::Brick brick{lower_bound, upper_bound,
                                    std::array<bool, 3>{{false, false, false}},
                                    num_elements, grid_points};
  test_brick_construction(
      brick, lower_bound, upper_bound, grid_points, num_elements,
      std::vector<std::unordered_map<Direction<3>, BlockNeighbor<3>>>{{}},
      std::vector<std::unordered_set<Direction<3>>>{
          {{Direction<3>::lower_xi()},
           {Direction<3>::upper_xi()},
           {Direction<3>::lower_eta()},
           {Direction<3>::upper_eta()},
           {Direction<3>::lower_zeta()},
           {Direction<3>::upper_zeta()}}});

  const DomainCreators::Brick periodic_x_brick{
      lower_bound, upper_bound, std::array<bool, 3>{{true, false, false}},
      num_elements, grid_points};
  test_brick_construction(
      periodic_x_brick, lower_bound, upper_bound, grid_points, num_elements,
      std::vector<std::unordered_map<Direction<3>, BlockNeighbor<3>>>{
          {{Direction<3>::lower_xi(), {0, aligned_orientation}},
           {Direction<3>::upper_xi(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{
          {{Direction<3>::lower_eta()},
           {Direction<3>::upper_eta()},
           {Direction<3>::lower_zeta()},
           {Direction<3>::upper_zeta()}}});

  const DomainCreators::Brick periodic_y_brick{
      lower_bound, upper_bound, std::array<bool, 3>{{false, true, false}},
      num_elements, grid_points};
  test_brick_construction(
      periodic_y_brick, lower_bound, upper_bound, grid_points, num_elements,
      std::vector<std::unordered_map<Direction<3>, BlockNeighbor<3>>>{
          {{Direction<3>::lower_eta(), {0, aligned_orientation}},
           {Direction<3>::upper_eta(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{
          {{Direction<3>::lower_xi()},
           {Direction<3>::upper_xi()},
           {Direction<3>::lower_zeta()},
           {Direction<3>::upper_zeta()}}});

  const DomainCreators::Brick periodic_z_brick{
      lower_bound, upper_bound, std::array<bool, 3>{{false, false, true}},
      num_elements, grid_points};
  test_brick_construction(
      periodic_z_brick, lower_bound, upper_bound, grid_points, num_elements,
      std::vector<std::unordered_map<Direction<3>, BlockNeighbor<3>>>{
          {{Direction<3>::lower_zeta(), {0, aligned_orientation}},
           {Direction<3>::upper_zeta(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{
          {{Direction<3>::lower_xi()},
           {Direction<3>::upper_xi()},
           {Direction<3>::lower_eta()},
           {Direction<3>::upper_eta()}}});

  const DomainCreators::Brick periodic_xy_brick{
      lower_bound, upper_bound, std::array<bool, 3>{{true, true, false}},
      num_elements, grid_points};
  test_brick_construction(
      periodic_xy_brick, lower_bound, upper_bound, grid_points, num_elements,
      std::vector<std::unordered_map<Direction<3>, BlockNeighbor<3>>>{
          {{Direction<3>::lower_xi(), {0, aligned_orientation}},
           {Direction<3>::upper_xi(), {0, aligned_orientation}},
           {Direction<3>::lower_eta(), {0, aligned_orientation}},
           {Direction<3>::upper_eta(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{
          {{Direction<3>::lower_zeta()}, {Direction<3>::upper_zeta()}}});

  const DomainCreators::Brick periodic_yz_brick{
      lower_bound, upper_bound, std::array<bool, 3>{{false, true, true}},
      num_elements, grid_points};
  test_brick_construction(
      periodic_yz_brick, lower_bound, upper_bound, grid_points, num_elements,
      std::vector<std::unordered_map<Direction<3>, BlockNeighbor<3>>>{
          {{Direction<3>::lower_eta(), {0, aligned_orientation}},
           {Direction<3>::upper_eta(), {0, aligned_orientation}},
           {Direction<3>::lower_zeta(), {0, aligned_orientation}},
           {Direction<3>::upper_zeta(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{{
          {Direction<3>::lower_xi()},
          {Direction<3>::upper_xi()},
      }});

  const DomainCreators::Brick periodic_xz_brick{
      lower_bound, upper_bound, std::array<bool, 3>{{true, false, true}},
      num_elements, grid_points};
  test_brick_construction(
      periodic_xz_brick, lower_bound, upper_bound, grid_points, num_elements,
      std::vector<std::unordered_map<Direction<3>, BlockNeighbor<3>>>{
          {{Direction<3>::lower_xi(), {0, aligned_orientation}},
           {Direction<3>::upper_xi(), {0, aligned_orientation}},
           {Direction<3>::lower_zeta(), {0, aligned_orientation}},
           {Direction<3>::upper_zeta(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{
          {{Direction<3>::lower_eta()}, {Direction<3>::upper_eta()}}});

  const DomainCreators::Brick periodic_xyz_brick{
      lower_bound, upper_bound, std::array<bool, 3>{{true, true, true}},
      num_elements, grid_points};
  test_brick_construction(
      periodic_xyz_brick, lower_bound, upper_bound, grid_points, num_elements,
      std::vector<std::unordered_map<Direction<3>, BlockNeighbor<3>>>{
          {{Direction<3>::lower_xi(), {0, aligned_orientation}},
           {Direction<3>::upper_xi(), {0, aligned_orientation}},
           {Direction<3>::lower_eta(), {0, aligned_orientation}},
           {Direction<3>::upper_eta(), {0, aligned_orientation}},
           {Direction<3>::lower_zeta(), {0, aligned_orientation}},
           {Direction<3>::upper_zeta(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{{}});
}

SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.Brick.Factory",
                  "[Domain][Unit]") {
  test_factory_creation<DomainCreator<3, Frame::Inertial>>(
      "  Brick:\n"
      "    LowerBound: [0,0,0]\n"
      "    UpperBound: [1,2,3]\n"
      "    IsPeriodicIn: [True,False,True]\n"
      "    InitialGridPoints: [3,4,3]\n"
      "    InitialRefinement: [2,3,2]\n");
}

// [[OutputRegex, index = 1]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.Brick.Extents",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  DomainCreators::Brick default_brick{};
  default_brick.initial_extents(1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
// [[OutputRegex, index = 2]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.Brick.Refinement",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  DomainCreators::Brick default_brick{};
  default_brick.initial_refinement_levels(2);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
