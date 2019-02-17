// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"          // IWYU pragma: keep
#include "Domain/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/OrientationMap.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/MakeVector.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "tests/Unit/Domain/DomainTestHelpers.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace domain {
namespace {
using Affine = CoordinateMaps::Affine;
using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
void test_brick_construction(
    const creators::Brick<Frame::Inertial>& brick,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound,
    const std::vector<std::array<size_t, 3>>& expected_extents,
    const std::vector<std::array<size_t, 3>>& expected_refinement_level,
    const std::vector<DirectionMap<3, BlockNeighbor<3>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<Direction<3>>>&
        expected_external_boundaries) {
  const auto domain = brick.create_domain();

  CHECK(brick.initial_extents() == expected_extents);
  CHECK(brick.initial_refinement_levels() == expected_refinement_level);

  test_domain_construction(
      domain, expected_block_neighbors, expected_external_boundaries,
      make_vector(make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Affine3D{Affine{-1., 1., lower_bound[0], upper_bound[0]},
                   Affine{-1., 1., lower_bound[1], upper_bound[1]},
                   Affine{-1., 1., lower_bound[2], upper_bound[2]}})));

  test_initial_domain(domain, brick.initial_refinement_levels());
}

void test_brick() {
  INFO("Brick");
  const std::vector<std::array<size_t, 3>> grid_points{{{4, 6, 3}}},
      refinement_level{{{3, 2, 4}}};
  const std::array<double, 3> lower_bound{{-1.2, 3.0, 2.5}},
      upper_bound{{0.8, 5.0, 3.0}};
  // Default OrientationMap is aligned.
  const OrientationMap<3> aligned_orientation{};

  const creators::Brick<Frame::Inertial> brick{
      lower_bound, upper_bound, std::array<bool, 3>{{false, false, false}},
      refinement_level[0], grid_points[0]};
  test_brick_construction(brick, lower_bound, upper_bound, grid_points,
                          refinement_level,
                          std::vector<DirectionMap<3, BlockNeighbor<3>>>{{}},
                          std::vector<std::unordered_set<Direction<3>>>{
                              {{Direction<3>::lower_xi()},
                               {Direction<3>::upper_xi()},
                               {Direction<3>::lower_eta()},
                               {Direction<3>::upper_eta()},
                               {Direction<3>::lower_zeta()},
                               {Direction<3>::upper_zeta()}}});

  const creators::Brick<Frame::Inertial> periodic_x_brick{
      lower_bound, upper_bound, std::array<bool, 3>{{true, false, false}},
      refinement_level[0], grid_points[0]};
  test_brick_construction(
      periodic_x_brick, lower_bound, upper_bound, grid_points, refinement_level,
      std::vector<DirectionMap<3, BlockNeighbor<3>>>{
          {{Direction<3>::lower_xi(), {0, aligned_orientation}},
           {Direction<3>::upper_xi(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{
          {{Direction<3>::lower_eta()},
           {Direction<3>::upper_eta()},
           {Direction<3>::lower_zeta()},
           {Direction<3>::upper_zeta()}}});

  const creators::Brick<Frame::Inertial> periodic_y_brick{
      lower_bound, upper_bound, std::array<bool, 3>{{false, true, false}},
      refinement_level[0], grid_points[0]};
  test_brick_construction(
      periodic_y_brick, lower_bound, upper_bound, grid_points, refinement_level,
      std::vector<DirectionMap<3, BlockNeighbor<3>>>{
          {{Direction<3>::lower_eta(), {0, aligned_orientation}},
           {Direction<3>::upper_eta(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{
          {{Direction<3>::lower_xi()},
           {Direction<3>::upper_xi()},
           {Direction<3>::lower_zeta()},
           {Direction<3>::upper_zeta()}}});

  const creators::Brick<Frame::Inertial> periodic_z_brick{
      lower_bound, upper_bound, std::array<bool, 3>{{false, false, true}},
      refinement_level[0], grid_points[0]};
  test_brick_construction(
      periodic_z_brick, lower_bound, upper_bound, grid_points, refinement_level,
      std::vector<DirectionMap<3, BlockNeighbor<3>>>{
          {{Direction<3>::lower_zeta(), {0, aligned_orientation}},
           {Direction<3>::upper_zeta(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{
          {{Direction<3>::lower_xi()},
           {Direction<3>::upper_xi()},
           {Direction<3>::lower_eta()},
           {Direction<3>::upper_eta()}}});

  const creators::Brick<Frame::Inertial> periodic_xy_brick{
      lower_bound, upper_bound, std::array<bool, 3>{{true, true, false}},
      refinement_level[0], grid_points[0]};
  test_brick_construction(
      periodic_xy_brick, lower_bound, upper_bound, grid_points,
      refinement_level,
      std::vector<DirectionMap<3, BlockNeighbor<3>>>{
          {{Direction<3>::lower_xi(), {0, aligned_orientation}},
           {Direction<3>::upper_xi(), {0, aligned_orientation}},
           {Direction<3>::lower_eta(), {0, aligned_orientation}},
           {Direction<3>::upper_eta(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{
          {{Direction<3>::lower_zeta()}, {Direction<3>::upper_zeta()}}});

  const creators::Brick<Frame::Inertial> periodic_yz_brick{
      lower_bound, upper_bound, std::array<bool, 3>{{false, true, true}},
      refinement_level[0], grid_points[0]};
  test_brick_construction(
      periodic_yz_brick, lower_bound, upper_bound, grid_points,
      refinement_level,
      std::vector<DirectionMap<3, BlockNeighbor<3>>>{
          {{Direction<3>::lower_eta(), {0, aligned_orientation}},
           {Direction<3>::upper_eta(), {0, aligned_orientation}},
           {Direction<3>::lower_zeta(), {0, aligned_orientation}},
           {Direction<3>::upper_zeta(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{{
          {Direction<3>::lower_xi()},
          {Direction<3>::upper_xi()},
      }});

  const creators::Brick<Frame::Inertial> periodic_xz_brick{
      lower_bound, upper_bound, std::array<bool, 3>{{true, false, true}},
      refinement_level[0], grid_points[0]};
  test_brick_construction(
      periodic_xz_brick, lower_bound, upper_bound, grid_points,
      refinement_level,
      std::vector<DirectionMap<3, BlockNeighbor<3>>>{
          {{Direction<3>::lower_xi(), {0, aligned_orientation}},
           {Direction<3>::upper_xi(), {0, aligned_orientation}},
           {Direction<3>::lower_zeta(), {0, aligned_orientation}},
           {Direction<3>::upper_zeta(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{
          {{Direction<3>::lower_eta()}, {Direction<3>::upper_eta()}}});

  const creators::Brick<Frame::Inertial> periodic_xyz_brick{
      lower_bound, upper_bound, std::array<bool, 3>{{true, true, true}},
      refinement_level[0], grid_points[0]};
  test_brick_construction(
      periodic_xyz_brick, lower_bound, upper_bound, grid_points,
      refinement_level,
      std::vector<DirectionMap<3, BlockNeighbor<3>>>{
          {{Direction<3>::lower_xi(), {0, aligned_orientation}},
           {Direction<3>::upper_xi(), {0, aligned_orientation}},
           {Direction<3>::lower_eta(), {0, aligned_orientation}},
           {Direction<3>::upper_eta(), {0, aligned_orientation}},
           {Direction<3>::lower_zeta(), {0, aligned_orientation}},
           {Direction<3>::upper_zeta(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{{}});

  // Test serialization of the map
  creators::register_derived_with_charm();

  const auto base_map =
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Affine3D{Affine{-1., 1., lower_bound[0], upper_bound[0]},
                   Affine{-1., 1., lower_bound[1], upper_bound[1]},
                   Affine{-1., 1., lower_bound[2], upper_bound[2]}});
  are_maps_equal(make_coordinate_map<Frame::Logical, Frame::Inertial>(
                     Affine3D{Affine{-1., 1., lower_bound[0], upper_bound[0]},
                              Affine{-1., 1., lower_bound[1], upper_bound[1]},
                              Affine{-1., 1., lower_bound[2], upper_bound[2]}}),
                 *serialize_and_deserialize(base_map));
}

void test_brick_factory() {
  INFO("Brick factory");
  const auto domain_creator =
      test_factory_creation<DomainCreator<3, Frame::Inertial>>(
          "  Brick:\n"
          "    LowerBound: [0,0,0]\n"
          "    UpperBound: [1,2,3]\n"
          "    IsPeriodicIn: [True,False,True]\n"
          "    InitialGridPoints: [3,4,3]\n"
          "    InitialRefinement: [2,3,2]\n");
  const auto* brick_creator =
      dynamic_cast<const creators::Brick<Frame::Inertial>*>(
          domain_creator.get());
  test_brick_construction(
      *brick_creator, {{0., 0., 0.}}, {{1., 2., 3.}}, {{{3, 4, 3}}},
      {{{2, 3, 2}}},
      std::vector<DirectionMap<3, BlockNeighbor<3>>>{
          {{Direction<3>::lower_xi(), {0, {}}},
           {Direction<3>::upper_xi(), {0, {}}},
           {Direction<3>::lower_zeta(), {0, {}}},
           {Direction<3>::upper_zeta(), {0, {}}}}},
      std::vector<std::unordered_set<Direction<3>>>{
          {{Direction<3>::lower_eta()}, {Direction<3>::upper_eta()}}});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.Brick", "[Domain][Unit]") {
  test_brick();
  test_brick_factory();
}
}  // namespace domain
