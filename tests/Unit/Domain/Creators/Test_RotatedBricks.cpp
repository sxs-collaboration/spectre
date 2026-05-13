// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Creators/RotatedBricks.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace domain {
namespace {
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_boundary_condition() {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::upper_zeta(), 2);
}

template <typename... FuncsOfTime>
void test_rotated_bricks_construction(
    const creators::RotatedBricks& rotated_bricks,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& midpoint,
    const std::array<double, 3>& upper_bound,
    const std::vector<std::array<size_t, 3>>& expected_extents,
    const std::vector<std::array<size_t, 3>>& expected_refinement_level,
    const std::vector<DirectionMap<3, BlockNeighbors<3>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<Direction<3>>>&
        expected_external_boundaries,
    const std::tuple<std::pair<std::string, FuncsOfTime>...>&
        expected_functions_of_time,
    const std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>&
        expected_grid_to_inertial_maps,
    const bool expect_boundary_conditions = false,
    const bool is_periodic = false,
    const std::unordered_map<std::string, double>& initial_expiration_times =
        {}) {
  const std::vector<double> times{1.};
  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      rotated_bricks, expect_boundary_conditions, is_periodic, times);
  CHECK(rotated_bricks.grid_anchors().empty());

  CHECK(rotated_bricks.initial_extents() == expected_extents);
  CHECK(rotated_bricks.initial_refinement_levels() ==
        expected_refinement_level);
  CHECK(rotated_bricks.block_names() ==
        make_vector("Block(0,0,0)"s, "Block(1,0,0)"s, "Block(0,1,0)"s,
                    "Block(1,1,0)"s, "Block(0,0,1)"s, "Block(1,0,1)"s,
                    "Block(0,1,1)"s, "Block(1,1,1)"s));

  using Interval = CoordinateMaps::Interval;
  using Interval3D =
      CoordinateMaps::ProductOf3Maps<Interval, Interval, Interval>;
  using DiscreteRotation3D = CoordinateMaps::DiscreteRotation<3>;
  using TargetFrame =
      tmpl::conditional_t<sizeof...(FuncsOfTime) == 0, Frame::Inertial,
                          Frame::Grid>;

  const Interval lower_x_map(-1.0, 1.0, lower_bound[0], midpoint[0],
                             CoordinateMaps::Distribution::Linear);
  const Interval upper_x_map(-1.0, 1.0, midpoint[0], upper_bound[0],
                             CoordinateMaps::Distribution::Linear);
  const Interval lower_y_map(-1.0, 1.0, lower_bound[1], midpoint[1],
                             CoordinateMaps::Distribution::Linear);
  const Interval upper_y_map(-1.0, 1.0, midpoint[1], upper_bound[1],
                             CoordinateMaps::Distribution::Linear);
  const Interval lower_z_map(-1.0, 1.0, lower_bound[2], midpoint[2],
                             CoordinateMaps::Distribution::Linear);
  const Interval upper_z_map(-1.0, 1.0, midpoint[2], upper_bound[2],
                             CoordinateMaps::Distribution::Linear);

  std::vector<std::unique_ptr<
      CoordinateMapBase<Frame::BlockLogical, TargetFrame, 3>>>
      coord_maps;
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
          Interval3D(lower_x_map, lower_y_map, lower_z_map)));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
          DiscreteRotation3D{OrientationMap<3>{std::array<Direction<3>, 3>{
              {Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
               Direction<3>::lower_xi()}}}},
          Interval3D(upper_x_map, lower_y_map, lower_z_map)));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
          DiscreteRotation3D{OrientationMap<3>{std::array<Direction<3>, 3>{
              {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
               Direction<3>::lower_eta()}}}},
          Interval3D(lower_x_map, upper_y_map, lower_z_map)));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
          DiscreteRotation3D{OrientationMap<3>{std::array<Direction<3>, 3>{
              {Direction<3>::upper_zeta(), Direction<3>::lower_xi(),
               Direction<3>::lower_eta()}}}},
          Interval3D(upper_x_map, upper_y_map, lower_z_map)));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
          DiscreteRotation3D{OrientationMap<3>{std::array<Direction<3>, 3>{
              {Direction<3>::upper_eta(), Direction<3>::lower_xi(),
               Direction<3>::upper_zeta()}}}},
          Interval3D(lower_x_map, lower_y_map, upper_z_map)));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
          DiscreteRotation3D{OrientationMap<3>{std::array<Direction<3>, 3>{
              {Direction<3>::upper_eta(), Direction<3>::lower_zeta(),
               Direction<3>::lower_xi()}}}},
          Interval3D(upper_x_map, lower_y_map, upper_z_map)));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
          DiscreteRotation3D{OrientationMap<3>{std::array<Direction<3>, 3>{
              {Direction<3>::upper_zeta(), Direction<3>::lower_xi(),
               Direction<3>::lower_eta()}}}},
          Interval3D(lower_x_map, upper_y_map, upper_z_map)));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
          Interval3D(upper_x_map, upper_y_map, upper_z_map)));

  test_domain_construction(domain, expected_block_neighbors,
                           expected_external_boundaries, coord_maps, 10.0,
                           rotated_bricks.functions_of_time(),
                           expected_grid_to_inertial_maps);
  TestHelpers::domain::creators::test_functions_of_time(
      rotated_bricks, expected_functions_of_time, initial_expiration_times);
}

void test_rotated_bricks() {
  INFO("Rotated bricks");
  const std::vector<std::array<size_t, 3>> grid_points{
      {{4, 2, 5}}, {{5, 2, 1}}, {{4, 5, 3}}, {{3, 5, 1}},
      {{2, 4, 6}}, {{6, 1, 2}}, {{3, 6, 4}}, {{1, 3, 6}}};
  const std::vector<std::array<size_t, 3>> refinement_level{
      {{0, 1, 2}}, {{2, 1, 0}}, {{0, 2, 1}}, {{1, 2, 0}},
      {{1, 0, 2}}, {{2, 0, 1}}, {{1, 2, 0}}, {{0, 1, 2}}};
  const std::array<double, 3> lower_bound{{-1.3, -3.0, 2.0}};
  const std::array<double, 3> midpoint{{-0.6, 0.3, 3.2}};
  const std::array<double, 3> upper_bound{{0.8, 3.0, 4.7}};
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
  for (const bool with_boundary_conditions : {true, false}) {
    CAPTURE(with_boundary_conditions);
    const creators::RotatedBricks rotated_bricks = [&]() {
      if (with_boundary_conditions) {
        return creators::RotatedBricks{
            lower_bound,
            midpoint,
            upper_bound,
            {{refinement_level[0][0], refinement_level[0][1],
              refinement_level[0][2]}},
            {{{{grid_points[0][0], grid_points[1][2]}},
              {{grid_points[0][1], grid_points[2][2]}},
              {{grid_points[0][2], grid_points[4][2]}}}},
            create_boundary_condition(),
            nullptr};
      } else {
        return creators::RotatedBricks{
            lower_bound,
            midpoint,
            upper_bound,
            {{refinement_level[0][0], refinement_level[0][1],
              refinement_level[0][2]}},
            {{{{grid_points[0][0], grid_points[1][2]}},
              {{grid_points[0][1], grid_points[2][2]}},
              {{grid_points[0][2], grid_points[4][2]}}}},
            {{false, false, false}},
            nullptr};
      }
    }();
    test_rotated_bricks_construction(
        rotated_bricks, lower_bound, midpoint, upper_bound, grid_points,
        refinement_level,
        std::vector<DirectionMap<3, BlockNeighbors<3>>>{
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
             {Direction<3>::lower_zeta(),
              {3, rotation_R_then_U.inverse_map()}}}},
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
             Direction<3>::upper_zeta()}},
        std::tuple<>{}, {}, with_boundary_conditions);

    const creators::RotatedBricks rotated_periodic_bricks = [&]() {
      if (with_boundary_conditions) {
        return creators::RotatedBricks{
            lower_bound,
            midpoint,
            upper_bound,
            {{refinement_level[0][0], refinement_level[0][1],
              refinement_level[0][2]}},
            {{{{grid_points[0][0], grid_points[1][2]}},
              {{grid_points[0][1], grid_points[2][2]}},
              {{grid_points[0][2], grid_points[4][2]}}}},
            std::make_unique<TestHelpers::domain::BoundaryConditions::
                                 TestPeriodicBoundaryCondition<3>>(),
            nullptr};
      } else {
        return creators::RotatedBricks{
            lower_bound,
            midpoint,
            upper_bound,
            {{refinement_level[0][0], refinement_level[0][1],
              refinement_level[0][2]}},
            {{{{grid_points[0][0], grid_points[1][2]}},
              {{grid_points[0][1], grid_points[2][2]}},
              {{grid_points[0][2], grid_points[4][2]}}}},
            {{true, true, true}},
            nullptr};
      }
    }();
    test_rotated_bricks_construction(
        rotated_periodic_bricks, lower_bound, midpoint, upper_bound,
        grid_points, refinement_level,
        std::vector<DirectionMap<3, BlockNeighbors<3>>>{
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
             {Direction<3>::lower_zeta(),
              {3, rotation_R_then_U.inverse_map()}}}},
        std::vector<std::unordered_set<Direction<3>>>{
            {}, {}, {}, {}, {}, {}, {}, {}},
        std::tuple<>{}, {}, with_boundary_conditions, true);
  }

  CHECK_THROWS_WITH(
      creators::RotatedBricks(
          lower_bound, midpoint, upper_bound,
          {{refinement_level[0][0], refinement_level[0][1],
            refinement_level[0][2]}},
          {{{{grid_points[0][0], grid_points[1][2]}},
            {{grid_points[0][1], grid_points[2][2]}},
            {{grid_points[0][2], grid_points[4][2]}}}},
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "None boundary condition is not supported. If you would like "
          "an outflow-type boundary condition, you must use that."));
}

void test_rotated_bricks_factory() {
  INFO("Rotated bricks factory");
  domain::creators::time_dependence::register_derived_with_charm();
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
  const std::vector<DirectionMap<3, BlockNeighbors<3>>>
      expected_block_neighbors_nonperiodic{
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
           {Direction<3>::lower_zeta(), {3, rotation_R_then_U.inverse_map()}}}};
  const std::vector<std::unordered_set<Direction<3>>>
      expected_external_boundaries_nonperiodic{
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
           Direction<3>::upper_zeta()}};
  const std::vector<std::array<size_t, 3>> expected_extents{
      {{3, 1, 5}}, {{5, 1, 2}}, {{3, 5, 4}}, {{4, 5, 2}},
      {{1, 3, 6}}, {{6, 2, 1}}, {{4, 6, 3}}, {{2, 4, 6}}};
  const std::vector<std::array<size_t, 3>> expected_refinement{
      {{2, 1, 0}}, {{0, 1, 2}}, {{2, 0, 1}}, {{1, 0, 2}},
      {{1, 2, 0}}, {{0, 2, 1}}, {{1, 0, 2}}, {{2, 1, 0}}};

  for (const bool with_boundary_conditions : {true, false}) {
    CAPTURE(with_boundary_conditions);
    const std::string opt_string{
        "RotatedBricks:\n"
        "  LowerBound: [0.1, -0.4, -0.2]\n"
        "  Midpoint:   [2.6, 3.2, 1.7]\n"
        "  UpperBound: [5.1, 6.2, 3.2]\n"
        "  InitialGridPoints: [[3,2],[1,4],[5,6]]\n"
        "  InitialRefinement: [2,1,0]\n" +
        std::string{with_boundary_conditions
                        ? "  BoundaryCondition:\n"
                          "    TestBoundaryCondition:\n"
                          "      Direction: upper-zeta\n"
                          "      BlockId: 2\n"
                        : "  IsPeriodicIn: [false, false, false]\n"} +
        "  TimeDependence: None\n"};
    const auto domain_creator = [&opt_string, with_boundary_conditions]() {
      if (with_boundary_conditions) {
        return TestHelpers::test_option_tag<
            domain::OptionTags::DomainCreator<3>,
            TestHelpers::domain::BoundaryConditions::
                MetavariablesWithBoundaryConditions<
                    3, domain::creators::RotatedBricks>>(opt_string);
      } else {
        return TestHelpers::test_option_tag<
            domain::OptionTags::DomainCreator<3>,
            TestHelpers::domain::BoundaryConditions::
                MetavariablesWithoutBoundaryConditions<
                    3, domain::creators::RotatedBricks>>(opt_string);
      }
    }();
    const auto* rotated_bricks_creator =
        dynamic_cast<const creators::RotatedBricks*>(domain_creator.get());
    test_rotated_bricks_construction(
        *rotated_bricks_creator, {{0.1, -0.4, -0.2}}, {{2.6, 3.2, 1.7}},
        {{5.1, 6.2, 3.2}}, expected_extents, expected_refinement,
        expected_block_neighbors_nonperiodic,
        expected_external_boundaries_nonperiodic, std::tuple<>{}, {},
        with_boundary_conditions);
  }
  {
    INFO("No boundary condition, time dependent");
    const auto domain_creator = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<3>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithoutBoundaryConditions<
                3, domain::creators::RotatedBricks>>(
        "RotatedBricks:\n"
        "  LowerBound: [0.1, -0.4, -0.2]\n"
        "  Midpoint:   [2.6, 3.2, 1.7]\n"
        "  UpperBound: [5.1, 6.2, 3.2]\n"
        "  InitialGridPoints: [[3,2],[1,4],[5,6]]\n"
        "  InitialRefinement: [2,1,0]\n"
        "  IsPeriodicIn: [false, false, false]\n"
        "  TimeDependence:\n"
        "    UniformTranslation:\n"
        "      InitialTime: 1.0\n"
        "      Velocity: [2.3, 0.5, 0.1]\n");
    const auto* rotated_bricks_creator =
        dynamic_cast<const creators::RotatedBricks*>(domain_creator.get());
    const double initial_time = 1.0;
    const DataVector velocity{{2.3, 0.5, 0.1}};
    const std::string f_of_t_name = "Translation";
    std::unordered_map<std::string, double> initial_expiration_times{};
    initial_expiration_times[f_of_t_name] = 10.0;
    // without expiration times
    test_rotated_bricks_construction(
        *rotated_bricks_creator, {{0.1, -0.4, -0.2}}, {{2.6, 3.2, 1.7}},
        {{5.1, 6.2, 3.2}}, expected_extents, expected_refinement,
        expected_block_neighbors_nonperiodic,
        expected_external_boundaries_nonperiodic,
        std::make_tuple(
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                f_of_t_name,
                {initial_time,
                 std::array<DataVector, 3>{
                     {{0.0, 0.0, 0.0}, velocity, {0.0, 0.0, 0.0}}},
                 std::numeric_limits<double>::infinity()}}),
        make_vector_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name}));
    // with expiration times
    test_rotated_bricks_construction(
        *rotated_bricks_creator, {{0.1, -0.4, -0.2}}, {{2.6, 3.2, 1.7}},
        {{5.1, 6.2, 3.2}}, expected_extents, expected_refinement,
        expected_block_neighbors_nonperiodic,
        expected_external_boundaries_nonperiodic,
        std::make_tuple(
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                f_of_t_name,
                {initial_time,
                 std::array<DataVector, 3>{
                     {{0.0, 0.0, 0.0}, velocity, {0.0, 0.0, 0.0}}},
                 initial_expiration_times[f_of_t_name]}}),
        make_vector_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name}),
        false, false, initial_expiration_times);
  }
  {
    INFO("With boundary condition, time dependent");
    const auto domain_creator = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<3>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithBoundaryConditions<
                3, domain::creators::RotatedBricks>>(
        "RotatedBricks:\n"
        "  LowerBound: [0.1, -0.4, -0.2]\n"
        "  Midpoint:   [2.6, 3.2, 1.7]\n"
        "  UpperBound: [5.1, 6.2, 3.2]\n"
        "  InitialGridPoints: [[3,2],[1,4],[5,6]]\n"
        "  InitialRefinement: [2,1,0]\n"
        "  BoundaryCondition:\n"
        "    TestBoundaryCondition:\n"
        "      Direction: upper-zeta\n"
        "      BlockId: 2\n"
        "  TimeDependence:\n"
        "    UniformTranslation:\n"
        "      InitialTime: 1.0\n"
        "      Velocity: [2.3, 0.5, 0.1]\n");
    const auto* rotated_bricks_creator =
        dynamic_cast<const creators::RotatedBricks*>(domain_creator.get());
    const double initial_time = 1.0;
    const DataVector velocity{{2.3, 0.5, 0.1}};
    const std::string f_of_t_name = "Translation";
    std::unordered_map<std::string, double> initial_expiration_times{};
    initial_expiration_times[f_of_t_name] = 10.0;
    // without expiration times
    test_rotated_bricks_construction(
        *rotated_bricks_creator, {{0.1, -0.4, -0.2}}, {{2.6, 3.2, 1.7}},
        {{5.1, 6.2, 3.2}}, expected_extents, expected_refinement,
        expected_block_neighbors_nonperiodic,
        expected_external_boundaries_nonperiodic,
        std::make_tuple(
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                f_of_t_name,
                {initial_time,
                 std::array<DataVector, 3>{
                     {{0.0, 0.0, 0.0}, velocity, {0.0, 0.0, 0.0}}},
                 std::numeric_limits<double>::infinity()}}),
        make_vector_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name}),
        true);
    // with expiration times
    test_rotated_bricks_construction(
        *rotated_bricks_creator, {{0.1, -0.4, -0.2}}, {{2.6, 3.2, 1.7}},
        {{5.1, 6.2, 3.2}}, expected_extents, expected_refinement,
        expected_block_neighbors_nonperiodic,
        expected_external_boundaries_nonperiodic,
        std::make_tuple(
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                f_of_t_name,
                {initial_time,
                 std::array<DataVector, 3>{
                     {{0.0, 0.0, 0.0}, velocity, {0.0, 0.0, 0.0}}},
                 initial_expiration_times[f_of_t_name]}}),
        make_vector_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<3>{f_of_t_name}),
        true, false, initial_expiration_times);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.RotatedBricks", "[Domain][Unit]") {
  test_rotated_bricks();
  test_rotated_bricks_factory();
}
}  // namespace domain
