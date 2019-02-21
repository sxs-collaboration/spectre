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
#include "Domain/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/RotatedBricks.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/OrientationMap.hpp"
#include "tests/Unit/Domain/DomainTestHelpers.hpp"
#include "tests/Unit/TestCreation.hpp"

namespace domain {
namespace {
void test_rotated_bricks_construction(
    const creators::RotatedBricks<Frame::Inertial>& rotated_bricks,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& midpoint,
    const std::array<double, 3>& upper_bound,
    const std::vector<std::array<size_t, 3>>& expected_extents,
    const std::vector<std::array<size_t, 3>>& expected_refinement_level,
    const std::vector<DirectionMap<3, BlockNeighbor<3>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<Direction<3>>>&
        expected_external_boundaries) noexcept {
  const auto domain = rotated_bricks.create_domain();

  CHECK(domain.blocks().size() == expected_extents.size());
  CHECK(domain.blocks().size() == expected_refinement_level.size());
  CHECK(rotated_bricks.initial_extents() == expected_extents);
  CHECK(rotated_bricks.initial_refinement_levels() ==
        expected_refinement_level);

  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using DiscreteRotation3D = CoordinateMaps::DiscreteRotation<3>;

  const Affine lower_x_map(-1.0, 1.0, lower_bound[0], midpoint[0]);
  const Affine upper_x_map(-1.0, 1.0, midpoint[0], upper_bound[0]);
  const Affine lower_y_map(-1.0, 1.0, lower_bound[1], midpoint[1]);
  const Affine upper_y_map(-1.0, 1.0, midpoint[1], upper_bound[1]);
  const Affine lower_z_map(-1.0, 1.0, lower_bound[2], midpoint[2]);
  const Affine upper_z_map(-1.0, 1.0, midpoint[2], upper_bound[2]);

  std::vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Inertial, 3>>>
      coord_maps;
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Affine3D(lower_x_map, lower_y_map, lower_z_map)));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          DiscreteRotation3D{OrientationMap<3>{std::array<Direction<3>, 3>{
              {Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
               Direction<3>::lower_xi()}}}},
          Affine3D(upper_x_map, lower_y_map, lower_z_map)));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          DiscreteRotation3D{OrientationMap<3>{std::array<Direction<3>, 3>{
              {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
               Direction<3>::lower_eta()}}}},
          Affine3D(lower_x_map, upper_y_map, lower_z_map)));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          DiscreteRotation3D{OrientationMap<3>{std::array<Direction<3>, 3>{
              {Direction<3>::upper_zeta(), Direction<3>::lower_xi(),
               Direction<3>::lower_eta()}}}},
          Affine3D(upper_x_map, upper_y_map, lower_z_map)));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          DiscreteRotation3D{OrientationMap<3>{std::array<Direction<3>, 3>{
              {Direction<3>::upper_eta(), Direction<3>::lower_xi(),
               Direction<3>::upper_zeta()}}}},
          Affine3D(lower_x_map, lower_y_map, upper_z_map)));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          DiscreteRotation3D{OrientationMap<3>{std::array<Direction<3>, 3>{
              {Direction<3>::upper_eta(), Direction<3>::lower_zeta(),
               Direction<3>::lower_xi()}}}},
          Affine3D(upper_x_map, lower_y_map, upper_z_map)));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          DiscreteRotation3D{OrientationMap<3>{std::array<Direction<3>, 3>{
              {Direction<3>::upper_zeta(), Direction<3>::lower_xi(),
               Direction<3>::lower_eta()}}}},
          Affine3D(lower_x_map, upper_y_map, upper_z_map)));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Affine3D(upper_x_map, upper_y_map, upper_z_map)));

  test_domain_construction(domain, expected_block_neighbors,
                           expected_external_boundaries, coord_maps);
  test_initial_domain(domain, rotated_bricks.initial_refinement_levels());
}

void test_rotated_bricks() {
  INFO("Rotated bricks");
  const std::vector<std::array<size_t, 3>> grid_points{
      {{4, 2, 5}}, {{5, 2, 1}}, {{4, 5, 3}}, {{3, 5, 1}},
      {{2, 4, 6}}, {{6, 1, 2}}, {{3, 6, 4}}, {{1, 3, 6}}},
      refinement_level{{{0, 1, 2}}, {{2, 1, 0}}, {{0, 2, 1}}, {{1, 2, 0}},
                       {{1, 0, 2}}, {{2, 0, 1}}, {{1, 2, 0}}, {{0, 1, 2}}};
  const std::array<double, 3> lower_bound{{-1.3, -3.0, 2.0}},
      midpoint{{-0.6, 0.3, 3.2}}, upper_bound{{0.8, 3.0, 4.7}};
  const OrientationMap<3> aligned{};
  const OrientationMap<3> rotation_F{std::array<Direction<3>, 3>{
      {Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
       Direction<3>::lower_xi()}}};
  const OrientationMap<3> rotation_R{std::array<Direction<3>, 3>{
      {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
       Direction<3>::lower_eta()}}};
  const OrientationMap<3> rotation_U{std::array<Direction<3>, 3>{
      {Direction<3>::upper_eta(), Direction<3>::lower_xi(),
       Direction<3>::upper_zeta()}}};
  const OrientationMap<3> rotation_R_then_U{std::array<Direction<3>, 3>{
      {Direction<3>::lower_eta(), Direction<3>::lower_zeta(),
       Direction<3>::upper_xi()}}};
  const OrientationMap<3> rotation_F_then_U{std::array<Direction<3>, 3>{
      {Direction<3>::lower_zeta(), Direction<3>::upper_xi(),
       Direction<3>::lower_eta()}}};
  const creators::RotatedBricks<Frame::Inertial> rotated_bricks{
      lower_bound,
      midpoint,
      upper_bound,
      {{false, false, false}},
      {{refinement_level[0][0], refinement_level[0][1],
        refinement_level[0][2]}},
      {{{{grid_points[0][0], grid_points[1][2]}},
        {{grid_points[0][1], grid_points[2][2]}},
        {{grid_points[0][2], grid_points[4][2]}}}}};
  test_rotated_bricks_construction(
      rotated_bricks, lower_bound, midpoint, upper_bound, grid_points,
      refinement_level,
      std::vector<DirectionMap<3, BlockNeighbor<3>>>{
          {{Direction<3>::upper_xi(), {1, rotation_F}},
           {Direction<3>::upper_eta(), {2, rotation_R}},
           {Direction<3>::upper_zeta(), {4, rotation_U}}},
          {{Direction<3>::lower_xi(), {5, rotation_R.inverse_map()}},
           {Direction<3>::upper_eta(), {3, rotation_U}},
           {Direction<3>::lower_zeta(), {0, rotation_F.inverse_map()}}},
          {{Direction<3>::upper_xi(), {3, rotation_F}},
           {Direction<3>::lower_eta(), {6, rotation_F}},
           {Direction<3>::lower_zeta(), {0, rotation_R.inverse_map()}}},
          {{Direction<3>::upper_xi(), {1, rotation_U.inverse_map()}},
           {Direction<3>::lower_eta(), {7, rotation_R_then_U}},
           {Direction<3>::lower_zeta(), {2, rotation_F.inverse_map()}}},
          {{Direction<3>::lower_xi(), {6, rotation_R}},
           {Direction<3>::upper_eta(), {5, rotation_F}},
           {Direction<3>::lower_zeta(), {0, rotation_U.inverse_map()}}},
          {{Direction<3>::upper_xi(), {1, rotation_R}},
           {Direction<3>::lower_eta(), {4, rotation_F.inverse_map()}},
           {Direction<3>::lower_zeta(), {7, rotation_F_then_U}}},
          {{Direction<3>::upper_xi(), {4, rotation_R.inverse_map()}},
           {Direction<3>::upper_eta(), {2, rotation_F.inverse_map()}},
           {Direction<3>::upper_zeta(), {7, rotation_R_then_U}}},
          {{Direction<3>::lower_xi(), {6, rotation_R_then_U.inverse_map()}},
           {Direction<3>::lower_eta(), {5, rotation_F_then_U.inverse_map()}},
           {Direction<3>::lower_zeta(), {3, rotation_R_then_U.inverse_map()}}}},
      std::vector<std::unordered_set<Direction<3>>>{
          {Direction<3>::lower_xi(), Direction<3>::lower_eta(),
           Direction<3>::lower_zeta()},
          {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
           Direction<3>::upper_zeta()},
          {Direction<3>::lower_xi(), Direction<3>::upper_eta(),
           Direction<3>::upper_zeta()},
          {Direction<3>::lower_xi(), Direction<3>::upper_eta(),
           Direction<3>::upper_zeta()},
          {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
           Direction<3>::upper_zeta()},
          {Direction<3>::lower_xi(), Direction<3>::upper_eta(),
           Direction<3>::upper_zeta()},
          {Direction<3>::lower_xi(), Direction<3>::lower_eta(),
           Direction<3>::lower_zeta()},
          {Direction<3>::upper_xi(), Direction<3>::upper_eta(),
           Direction<3>::upper_zeta()}});
  test_physical_separation(rotated_bricks.create_domain().blocks());

  const creators::RotatedBricks<Frame::Inertial> rotated_periodic_bricks{
      lower_bound,
      midpoint,
      upper_bound,
      {{true, true, true}},
      {{refinement_level[0][0], refinement_level[0][1],
        refinement_level[0][2]}},
      {{{{grid_points[0][0], grid_points[1][2]}},
        {{grid_points[0][1], grid_points[2][2]}},
        {{grid_points[0][2], grid_points[4][2]}}}}};
  test_rotated_bricks_construction(
      rotated_periodic_bricks, lower_bound, midpoint, upper_bound, grid_points,
      refinement_level,
      std::vector<DirectionMap<3, BlockNeighbor<3>>>{
          {{Direction<3>::upper_xi(), {1, rotation_F}},
           {Direction<3>::upper_eta(), {2, rotation_R}},
           {Direction<3>::upper_zeta(), {4, rotation_U}},
           {Direction<3>::lower_xi(), {1, rotation_F}},
           {Direction<3>::lower_eta(), {2, rotation_R}},
           {Direction<3>::lower_zeta(), {4, rotation_U}}},
          {{Direction<3>::lower_xi(), {5, rotation_R.inverse_map()}},
           {Direction<3>::upper_eta(), {3, rotation_U}},
           {Direction<3>::lower_zeta(), {0, rotation_F.inverse_map()}},
           {Direction<3>::upper_xi(), {5, rotation_R.inverse_map()}},
           {Direction<3>::lower_eta(), {3, rotation_U}},
           {Direction<3>::upper_zeta(), {0, rotation_F.inverse_map()}}},
          {{Direction<3>::upper_xi(), {3, rotation_F}},
           {Direction<3>::lower_eta(), {6, rotation_F}},
           {Direction<3>::lower_zeta(), {0, rotation_R.inverse_map()}},
           {Direction<3>::lower_xi(), {3, rotation_F}},
           {Direction<3>::upper_eta(), {6, rotation_F}},
           {Direction<3>::upper_zeta(), {0, rotation_R.inverse_map()}}},
          {{Direction<3>::upper_xi(), {1, rotation_U.inverse_map()}},
           {Direction<3>::lower_eta(), {7, rotation_R_then_U}},
           {Direction<3>::lower_zeta(), {2, rotation_F.inverse_map()}},
           {Direction<3>::lower_xi(), {1, rotation_U.inverse_map()}},
           {Direction<3>::upper_eta(), {7, rotation_R_then_U}},
           {Direction<3>::upper_zeta(), {2, rotation_F.inverse_map()}}},
          {{Direction<3>::lower_xi(), {6, rotation_R}},
           {Direction<3>::upper_eta(), {5, rotation_F}},
           {Direction<3>::lower_zeta(), {0, rotation_U.inverse_map()}},
           {Direction<3>::upper_xi(), {6, rotation_R}},
           {Direction<3>::lower_eta(), {5, rotation_F}},
           {Direction<3>::upper_zeta(), {0, rotation_U.inverse_map()}}},
          {{Direction<3>::upper_xi(), {1, rotation_R}},
           {Direction<3>::lower_eta(), {4, rotation_F.inverse_map()}},
           {Direction<3>::lower_zeta(), {7, rotation_F_then_U}},
           {Direction<3>::lower_xi(), {1, rotation_R}},
           {Direction<3>::upper_eta(), {4, rotation_F.inverse_map()}},
           {Direction<3>::upper_zeta(), {7, rotation_F_then_U}}},
          {{Direction<3>::upper_xi(), {4, rotation_R.inverse_map()}},
           {Direction<3>::upper_eta(), {2, rotation_F.inverse_map()}},
           {Direction<3>::upper_zeta(), {7, rotation_R_then_U}},
           {Direction<3>::lower_xi(), {4, rotation_R.inverse_map()}},
           {Direction<3>::lower_eta(), {2, rotation_F.inverse_map()}},
           {Direction<3>::lower_zeta(), {7, rotation_R_then_U}}},
          {{Direction<3>::upper_xi(), {6, rotation_R_then_U.inverse_map()}},
           {Direction<3>::upper_eta(), {5, rotation_F_then_U.inverse_map()}},
           {Direction<3>::upper_zeta(), {3, rotation_R_then_U.inverse_map()}},
           {Direction<3>::lower_xi(), {6, rotation_R_then_U.inverse_map()}},
           {Direction<3>::lower_eta(), {5, rotation_F_then_U.inverse_map()}},
           {Direction<3>::lower_zeta(), {3, rotation_R_then_U.inverse_map()}}}},
      std::vector<std::unordered_set<Direction<3>>>{
          {}, {}, {}, {}, {}, {}, {}, {}});
}

void test_rotated_bricks_factory() {
  INFO("Rotated bricks factory");
  const OrientationMap<3> aligned{};
  const OrientationMap<3> rotation_F{std::array<Direction<3>, 3>{
      {Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
       Direction<3>::lower_xi()}}};
  const OrientationMap<3> rotation_R{std::array<Direction<3>, 3>{
      {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
       Direction<3>::lower_eta()}}};
  const OrientationMap<3> rotation_U{std::array<Direction<3>, 3>{
      {Direction<3>::upper_eta(), Direction<3>::lower_xi(),
       Direction<3>::upper_zeta()}}};
  const OrientationMap<3> rotation_R_then_U{std::array<Direction<3>, 3>{
      {Direction<3>::lower_eta(), Direction<3>::lower_zeta(),
       Direction<3>::upper_xi()}}};
  const OrientationMap<3> rotation_F_then_U{std::array<Direction<3>, 3>{
      {Direction<3>::lower_zeta(), Direction<3>::upper_xi(),
       Direction<3>::lower_eta()}}};
  const auto domain_creator =
      test_factory_creation<DomainCreator<3, Frame::Inertial>>(
          "  RotatedBricks:\n"
          "    LowerBound: [0.1, -0.4, -0.2]\n"
          "    Midpoint:   [2.6, 3.2, 1.7]\n"
          "    UpperBound: [5.1, 6.2, 3.2]\n"
          "    IsPeriodicIn: [false, false, false]\n"
          "    InitialGridPoints: [[3,2],[1,4],[5,6]]\n"
          "    InitialRefinement: [2,1,0]\n");
  const auto* rotated_bricks_creator =
      dynamic_cast<const creators::RotatedBricks<Frame::Inertial>*>(
          domain_creator.get());
  test_rotated_bricks_construction(
      *rotated_bricks_creator, {{0.1, -0.4, -0.2}}, {{2.6, 3.2, 1.7}},
      {{5.1, 6.2, 3.2}},
      {{{3, 1, 5}},
       {{5, 1, 2}},
       {{3, 5, 4}},
       {{4, 5, 2}},
       {{1, 3, 6}},
       {{6, 2, 1}},
       {{4, 6, 3}},
       {{2, 4, 6}}},
      {{{2, 1, 0}},
       {{0, 1, 2}},
       {{2, 0, 1}},
       {{1, 0, 2}},
       {{1, 2, 0}},
       {{0, 2, 1}},
       {{1, 0, 2}},
       {{2, 1, 0}}},
      std::vector<DirectionMap<3, BlockNeighbor<3>>>{
          {{Direction<3>::upper_xi(), {1, rotation_F}},
           {Direction<3>::upper_eta(), {2, rotation_R}},
           {Direction<3>::upper_zeta(), {4, rotation_U}}},
          {{Direction<3>::lower_xi(), {5, rotation_R.inverse_map()}},
           {Direction<3>::upper_eta(), {3, rotation_U}},
           {Direction<3>::lower_zeta(), {0, rotation_F.inverse_map()}}},
          {{Direction<3>::upper_xi(), {3, rotation_F}},
           {Direction<3>::lower_eta(), {6, rotation_F}},
           {Direction<3>::lower_zeta(), {0, rotation_R.inverse_map()}}},
          {{Direction<3>::upper_xi(), {1, rotation_U.inverse_map()}},
           {Direction<3>::lower_eta(), {7, rotation_R_then_U}},
           {Direction<3>::lower_zeta(), {2, rotation_F.inverse_map()}}},
          {{Direction<3>::lower_xi(), {6, rotation_R}},
           {Direction<3>::upper_eta(), {5, rotation_F}},
           {Direction<3>::lower_zeta(), {0, rotation_U.inverse_map()}}},
          {{Direction<3>::upper_xi(), {1, rotation_R}},
           {Direction<3>::lower_eta(), {4, rotation_F.inverse_map()}},
           {Direction<3>::lower_zeta(), {7, rotation_F_then_U}}},
          {{Direction<3>::upper_xi(), {4, rotation_R.inverse_map()}},
           {Direction<3>::upper_eta(), {2, rotation_F.inverse_map()}},
           {Direction<3>::upper_zeta(), {7, rotation_R_then_U}}},
          {{Direction<3>::lower_xi(), {6, rotation_R_then_U.inverse_map()}},
           {Direction<3>::lower_eta(), {5, rotation_F_then_U.inverse_map()}},
           {Direction<3>::lower_zeta(), {3, rotation_R_then_U.inverse_map()}}}},
      std::vector<std::unordered_set<Direction<3>>>{
          {Direction<3>::lower_xi(), Direction<3>::lower_eta(),
           Direction<3>::lower_zeta()},
          {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
           Direction<3>::upper_zeta()},
          {Direction<3>::lower_xi(), Direction<3>::upper_eta(),
           Direction<3>::upper_zeta()},
          {Direction<3>::lower_xi(), Direction<3>::upper_eta(),
           Direction<3>::upper_zeta()},
          {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
           Direction<3>::upper_zeta()},
          {Direction<3>::lower_xi(), Direction<3>::upper_eta(),
           Direction<3>::upper_zeta()},
          {Direction<3>::lower_xi(), Direction<3>::lower_eta(),
           Direction<3>::lower_zeta()},
          {Direction<3>::upper_xi(), Direction<3>::upper_eta(),
           Direction<3>::upper_zeta()}});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.RotatedBricks.Factory",
                  "[Domain][Unit]") {
  test_rotated_bricks();
  test_rotated_bricks_factory();
}
}  // namespace domain
