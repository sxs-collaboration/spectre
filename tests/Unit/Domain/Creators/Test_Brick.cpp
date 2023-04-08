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
#include "Domain/Block.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace domain {
namespace {
using Affine = CoordinateMaps::Affine;
using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
using Translation3D = CoordinateMaps::TimeDependent::Translation<3>;

template <typename... FuncsOfTime>
void test_brick_construction(
    const creators::Brick& brick, const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound,
    const std::vector<std::array<size_t, 3>>& expected_extents,
    const std::vector<std::array<size_t, 3>>& expected_refinement_level,
    const std::vector<DirectionMap<3, BlockNeighbor<3>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<Direction<3>>>&
        expected_external_boundaries,
    const std::tuple<std::pair<std::string, FuncsOfTime>...>&
        expected_functions_of_time = {},
    const std::vector<std::unique_ptr<domain::CoordinateMapBase<
        Frame::Grid, Frame::Inertial, 3>>>& expected_grid_to_inertial_maps = {},
    const bool expect_boundary_conditions = false,
    const std::unordered_map<std::string, double>& initial_expiration_times =
        {}) {
  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      brick, expect_boundary_conditions);

  CHECK(brick.initial_extents() == expected_extents);
  CHECK(brick.initial_refinement_levels() == expected_refinement_level);
  CHECK(brick.block_names() == std::vector<std::string>{"Brick"});
  CHECK(brick.block_groups().empty());

  test_domain_construction(
      domain, expected_block_neighbors, expected_external_boundaries,
      make_vector(make_coordinate_map_base<
                  Frame::BlockLogical,
                  tmpl::conditional_t<sizeof...(FuncsOfTime) == 0,
                                      Frame::Inertial, Frame::Grid>>(
          Affine3D{Affine{-1., 1., lower_bound[0], upper_bound[0]},
                   Affine{-1., 1., lower_bound[1], upper_bound[1]},
                   Affine{-1., 1., lower_bound[2], upper_bound[2]}})),
      10.0, brick.functions_of_time(), expected_grid_to_inertial_maps);

  TestHelpers::domain::creators::test_functions_of_time(
      brick, expected_functions_of_time, initial_expiration_times);
}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_boundary_condition() {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::upper_zeta(), 2);
}

void test_brick() {
  INFO("Brick");
  const std::vector<std::array<size_t, 3>> grid_points{{{4, 6, 3}}};
  const std::vector<std::array<size_t, 3>> refinement_level{{{3, 2, 4}}};
  const std::array<double, 3> lower_bound{{-1.2, 3.0, 2.5}};
  const std::array<double, 3> upper_bound{{0.8, 5.0, 3.0}};
  // Default OrientationMap is aligned.
  const OrientationMap<3> aligned_orientation{};

  const creators::Brick brick{lower_bound, upper_bound, refinement_level[0],
                              grid_points[0],
                              std::array<bool, 3>{{false, false, false}}};
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

  const creators::Brick brick_boundary_condition{lower_bound,
                                                 upper_bound,
                                                 refinement_level[0],
                                                 grid_points[0],
                                                 create_boundary_condition(),
                                                 create_boundary_condition(),
                                                 create_boundary_condition(),
                                                 {}};
  test_brick_construction(brick_boundary_condition, lower_bound, upper_bound,
                          grid_points, refinement_level,
                          std::vector<DirectionMap<3, BlockNeighbor<3>>>{{}},
                          std::vector<std::unordered_set<Direction<3>>>{
                              {{Direction<3>::lower_xi()},
                               {Direction<3>::upper_xi()},
                               {Direction<3>::lower_eta()},
                               {Direction<3>::upper_eta()},
                               {Direction<3>::lower_zeta()},
                               {Direction<3>::upper_zeta()}}},
                          {}, {}, true);

  const creators::Brick periodic_x_brick{
      lower_bound, upper_bound, refinement_level[0], grid_points[0],
      std::array<bool, 3>{{true, false, false}}};
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

  const creators::Brick periodic_y_brick{
      lower_bound, upper_bound, refinement_level[0], grid_points[0],
      std::array<bool, 3>{{false, true, false}}};
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

  // Test periodic in z
  test_brick_construction(
      creators::Brick{lower_bound, upper_bound, refinement_level[0],
                      grid_points[0],
                      std::array<bool, 3>{{false, false, true}}},
      lower_bound, upper_bound, grid_points, refinement_level,
      std::vector<DirectionMap<3, BlockNeighbor<3>>>{
          {{Direction<3>::lower_zeta(), {0, aligned_orientation}},
           {Direction<3>::upper_zeta(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{
          {{Direction<3>::lower_xi()},
           {Direction<3>::upper_xi()},
           {Direction<3>::lower_eta()},
           {Direction<3>::upper_eta()}}});
  test_brick_construction(
      creators::Brick{lower_bound, upper_bound, refinement_level[0],
                      grid_points[0], create_boundary_condition(),
                      create_boundary_condition(),
                      TestHelpers::domain::BoundaryConditions::
                          TestPeriodicBoundaryCondition<3>{}
                              .get_clone()},
      lower_bound, upper_bound, grid_points, refinement_level,
      std::vector<DirectionMap<3, BlockNeighbor<3>>>{
          {{Direction<3>::lower_zeta(), {0, aligned_orientation}},
           {Direction<3>::upper_zeta(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{
          {{Direction<3>::lower_xi()},
           {Direction<3>::upper_xi()},
           {Direction<3>::lower_eta()},
           {Direction<3>::upper_eta()}}},
      {}, {}, true);
  // Periodic in x
  test_brick_construction(
      creators::Brick{lower_bound, upper_bound, refinement_level[0],
                      grid_points[0],
                      TestHelpers::domain::BoundaryConditions::
                          TestPeriodicBoundaryCondition<3>{}
                              .get_clone(),
                      create_boundary_condition(), create_boundary_condition()},
      lower_bound, upper_bound, grid_points, refinement_level,
      std::vector<DirectionMap<3, BlockNeighbor<3>>>{
          {{Direction<3>::lower_xi(), {0, aligned_orientation}},
           {Direction<3>::upper_xi(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{
          {{Direction<3>::lower_eta()},
           {Direction<3>::upper_eta()},
           {Direction<3>::lower_zeta()},
           {Direction<3>::upper_zeta()}}},
      {}, {}, true);
  // Periodic in y
  test_brick_construction(
      creators::Brick{lower_bound, upper_bound, refinement_level[0],
                      grid_points[0], create_boundary_condition(),
                      TestHelpers::domain::BoundaryConditions::
                          TestPeriodicBoundaryCondition<3>{}
                              .get_clone(),
                      create_boundary_condition()},
      lower_bound, upper_bound, grid_points, refinement_level,
      std::vector<DirectionMap<3, BlockNeighbor<3>>>{
          {{Direction<3>::lower_eta(), {0, aligned_orientation}},
           {Direction<3>::upper_eta(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{
          {{Direction<3>::lower_xi()},
           {Direction<3>::upper_xi()},
           {Direction<3>::lower_zeta()},
           {Direction<3>::upper_zeta()}}},
      {}, {}, true);

  // Test periodic in xy
  test_brick_construction(
      creators::Brick{lower_bound, upper_bound, refinement_level[0],
                      grid_points[0], std::array<bool, 3>{{true, true, false}}},
      lower_bound, upper_bound, grid_points, refinement_level,
      std::vector<DirectionMap<3, BlockNeighbor<3>>>{
          {{Direction<3>::lower_xi(), {0, aligned_orientation}},
           {Direction<3>::upper_xi(), {0, aligned_orientation}},
           {Direction<3>::lower_eta(), {0, aligned_orientation}},
           {Direction<3>::upper_eta(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{
          {{Direction<3>::lower_zeta()}, {Direction<3>::upper_zeta()}}});
  test_brick_construction(
      creators::Brick{lower_bound, upper_bound, refinement_level[0],
                      grid_points[0],
                      TestHelpers::domain::BoundaryConditions::
                          TestPeriodicBoundaryCondition<3>{}
                              .get_clone(),
                      TestHelpers::domain::BoundaryConditions::
                          TestPeriodicBoundaryCondition<3>{}
                              .get_clone(),
                      create_boundary_condition()},
      lower_bound, upper_bound, grid_points, refinement_level,
      std::vector<DirectionMap<3, BlockNeighbor<3>>>{
          {{Direction<3>::lower_xi(), {0, aligned_orientation}},
           {Direction<3>::upper_xi(), {0, aligned_orientation}},
           {Direction<3>::lower_eta(), {0, aligned_orientation}},
           {Direction<3>::upper_eta(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{
          {{Direction<3>::lower_zeta()}, {Direction<3>::upper_zeta()}}},
      {}, {}, true);

  // Test periodic in yz
  const creators::Brick periodic_yz_brick{
      lower_bound, upper_bound, refinement_level[0], grid_points[0],
      std::array<bool, 3>{{false, true, true}}};
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
  test_brick_construction(
      creators::Brick{lower_bound, upper_bound, refinement_level[0],
                      grid_points[0], create_boundary_condition(),
                      TestHelpers::domain::BoundaryConditions::
                          TestPeriodicBoundaryCondition<3>{}
                              .get_clone(),
                      TestHelpers::domain::BoundaryConditions::
                          TestPeriodicBoundaryCondition<3>{}
                              .get_clone()},
      lower_bound, upper_bound, grid_points, refinement_level,
      std::vector<DirectionMap<3, BlockNeighbor<3>>>{
          {{Direction<3>::lower_eta(), {0, aligned_orientation}},
           {Direction<3>::upper_eta(), {0, aligned_orientation}},
           {Direction<3>::lower_zeta(), {0, aligned_orientation}},
           {Direction<3>::upper_zeta(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{
          {{Direction<3>::lower_xi()}, {Direction<3>::upper_xi()}}},
      {}, {}, true);

  const creators::Brick periodic_xz_brick{
      lower_bound, upper_bound, refinement_level[0], grid_points[0],
      std::array<bool, 3>{{true, false, true}}};
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
  test_brick_construction(
      creators::Brick{lower_bound, upper_bound, refinement_level[0],
                      grid_points[0],
                      TestHelpers::domain::BoundaryConditions::
                          TestPeriodicBoundaryCondition<3>{}
                              .get_clone(),
                      create_boundary_condition(),
                      TestHelpers::domain::BoundaryConditions::
                          TestPeriodicBoundaryCondition<3>{}
                              .get_clone()},
      lower_bound, upper_bound, grid_points, refinement_level,
      std::vector<DirectionMap<3, BlockNeighbor<3>>>{
          {{Direction<3>::lower_xi(), {0, aligned_orientation}},
           {Direction<3>::upper_xi(), {0, aligned_orientation}},
           {Direction<3>::lower_zeta(), {0, aligned_orientation}},
           {Direction<3>::upper_zeta(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{
          {{Direction<3>::lower_eta()}, {Direction<3>::upper_eta()}}},
      {}, {}, true);

  const creators::Brick periodic_xyz_brick{
      lower_bound, upper_bound, refinement_level[0], grid_points[0],
      std::array<bool, 3>{{true, true, true}}};
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

  const creators::Brick periodic_brick_boundary_conditions{
      lower_bound,
      upper_bound,
      refinement_level[0],
      grid_points[0],
      TestHelpers::domain::BoundaryConditions::TestPeriodicBoundaryCondition<
          3>{}
          .get_clone(),
      TestHelpers::domain::BoundaryConditions::TestPeriodicBoundaryCondition<
          3>{}
          .get_clone(),
      TestHelpers::domain::BoundaryConditions::TestPeriodicBoundaryCondition<
          3>{}
          .get_clone()};
  test_brick_construction(
      periodic_brick_boundary_conditions, lower_bound, upper_bound, grid_points,
      refinement_level,
      std::vector<DirectionMap<3, BlockNeighbor<3>>>{
          {{Direction<3>::lower_xi(), {0, aligned_orientation}},
           {Direction<3>::upper_xi(), {0, aligned_orientation}},
           {Direction<3>::lower_eta(), {0, aligned_orientation}},
           {Direction<3>::upper_eta(), {0, aligned_orientation}},
           {Direction<3>::lower_zeta(), {0, aligned_orientation}},
           {Direction<3>::upper_zeta(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<3>>>{{}}, {}, {}, true);

  // Test serialization of the map
  creators::register_derived_with_charm();

  const auto base_map =
      make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          Affine3D{Affine{-1., 1., lower_bound[0], upper_bound[0]},
                   Affine{-1., 1., lower_bound[1], upper_bound[1]},
                   Affine{-1., 1., lower_bound[2], upper_bound[2]}});
  are_maps_equal(make_coordinate_map<Frame::BlockLogical, Frame::Inertial>(
                     Affine3D{Affine{-1., 1., lower_bound[0], upper_bound[0]},
                              Affine{-1., 1., lower_bound[1], upper_bound[1]},
                              Affine{-1., 1., lower_bound[2], upper_bound[2]}}),
                 *serialize_and_deserialize(base_map));

  CHECK_THROWS_WITH(
      creators::Brick(lower_bound, upper_bound, refinement_level[0],
                      grid_points[0],
                      std::make_unique<TestHelpers::domain::BoundaryConditions::
                                           TestNoneBoundaryCondition<3>>(),
                      std::make_unique<TestHelpers::domain::BoundaryConditions::
                                           TestPeriodicBoundaryCondition<3>>(),
                      std::make_unique<TestHelpers::domain::BoundaryConditions::
                                           TestPeriodicBoundaryCondition<3>>(),
                      nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "None boundary condition is not supported. If you would like an "
          "outflow-type boundary condition, you must use that."));
  CHECK_THROWS_WITH(
      creators::Brick(lower_bound, upper_bound, refinement_level[0],
                      grid_points[0],
                      std::make_unique<TestHelpers::domain::BoundaryConditions::
                                           TestPeriodicBoundaryCondition<3>>(),
                      std::make_unique<TestHelpers::domain::BoundaryConditions::
                                           TestNoneBoundaryCondition<3>>(),
                      std::make_unique<TestHelpers::domain::BoundaryConditions::
                                           TestPeriodicBoundaryCondition<3>>(),
                      nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "None boundary condition is not supported. If you would like an "
          "outflow-type boundary condition, you must use that."));
  CHECK_THROWS_WITH(
      creators::Brick(lower_bound, upper_bound, refinement_level[0],
                      grid_points[0],
                      std::make_unique<TestHelpers::domain::BoundaryConditions::
                                           TestPeriodicBoundaryCondition<3>>(),
                      std::make_unique<TestHelpers::domain::BoundaryConditions::
                                           TestPeriodicBoundaryCondition<3>>(),
                      std::make_unique<TestHelpers::domain::BoundaryConditions::
                                           TestNoneBoundaryCondition<3>>(),
                      nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "None boundary condition is not supported. If you would like an "
          "outflow-type boundary condition, you must use that."));
}

void test_brick_factory() {
  const std::string boundary_conditions{
      "  BoundaryConditionInX:\n"
      "    TestBoundaryCondition:\n"
      "      Direction: upper-zeta\n"
      "      BlockId: 2\n"
      "  BoundaryConditionInY:\n"
      "    TestBoundaryCondition:\n"
      "      Direction: upper-zeta\n"
      "      BlockId: 2\n"
      "  BoundaryConditionInZ:\n"
      "    TestBoundaryCondition:\n"
      "      Direction: upper-zeta\n"
      "      BlockId: 2\n"};
  {
    INFO("Brick factory time independent, no boundary condition");
    const auto domain_creator = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<3>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithoutBoundaryConditions<3, domain::creators::Brick>>(
        "Brick:\n"
        "  LowerBound: [0,0,0]\n"
        "  UpperBound: [1,2,3]\n"
        "  IsPeriodicIn: [True,False,True]\n"
        "  InitialGridPoints: [3,4,3]\n"
        "  InitialRefinement: [2,3,2]\n"
        "  TimeDependence: None\n");
    const auto* brick_creator =
        dynamic_cast<const creators::Brick*>(domain_creator.get());
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
  {
    INFO("Brick factory time independent, with boundary condition");
    const auto domain_creator = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<3>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithBoundaryConditions<3, domain::creators::Brick>>(
        "Brick:\n"
        "  LowerBound: [0,0,0]\n"
        "  UpperBound: [1,2,3]\n"
        "  InitialGridPoints: [3,4,3]\n"
        "  InitialRefinement: [2,3,2]\n"
        "  TimeDependence: None\n" +
        boundary_conditions);
    const auto* brick_creator =
        dynamic_cast<const creators::Brick*>(domain_creator.get());
    test_brick_construction(*brick_creator, {{0., 0., 0.}}, {{1., 2., 3.}},
                            {{{3, 4, 3}}}, {{{2, 3, 2}}}, {{}},
                            std::vector<std::unordered_set<Direction<3>>>{
                                {{Direction<3>::lower_xi()},
                                 {Direction<3>::upper_xi()},
                                 {Direction<3>::lower_eta()},
                                 {Direction<3>::upper_eta()},
                                 {Direction<3>::lower_zeta()},
                                 {Direction<3>::upper_zeta()}}},
                            {}, {}, true);
  }
  {
    INFO("Brick factory time dependent");
    const auto domain_creator = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<3>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithoutBoundaryConditions<3, domain::creators::Brick>>(
        "Brick:\n"
        "  LowerBound: [0,0,0]\n"
        "  UpperBound: [1,2,3]\n"
        "  IsPeriodicIn: [True,False,True]\n"
        "  InitialGridPoints: [3,4,3]\n"
        "  InitialRefinement: [2,3,2]\n"
        "  TimeDependence:\n"
        "    UniformTranslation:\n"
        "      InitialTime: 1.0\n"
        "      Velocity: [2.3, -0.3, 0.5]\n");
    const auto* brick_creator =
        dynamic_cast<const creators::Brick*>(domain_creator.get());
    const double initial_time = 1.0;
    const DataVector velocity{{2.3, -0.3, 0.5}};
    // This name must match the hard coded one in UniformTranslation
    const std::string f_of_t_name = "Translation";
    std::unordered_map<std::string, double> initial_expiration_times{};
    initial_expiration_times[f_of_t_name] = 10.0;
    // without expiration times
    test_brick_construction(
        *brick_creator, {{0., 0., 0.}}, {{1., 2., 3.}}, {{{3, 4, 3}}},
        {{{2, 3, 2}}},
        std::vector<DirectionMap<3, BlockNeighbor<3>>>{
            {{Direction<3>::lower_xi(), {0, {}}},
             {Direction<3>::upper_xi(), {0, {}}},
             {Direction<3>::lower_zeta(), {0, {}}},
             {Direction<3>::upper_zeta(), {0, {}}}}},
        std::vector<std::unordered_set<Direction<3>>>{
            {{Direction<3>::lower_eta()}, {Direction<3>::upper_eta()}}},
        std::make_tuple(
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                f_of_t_name,
                {initial_time,
                 std::array<DataVector, 3>{{{3, 0.0}, velocity, {3, 0.0}}},
                 std::numeric_limits<double>::infinity()}}),
        make_vector_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            Translation3D{f_of_t_name}));
    // with expiration times
    test_brick_construction(
        *brick_creator, {{0., 0., 0.}}, {{1., 2., 3.}}, {{{3, 4, 3}}},
        {{{2, 3, 2}}},
        std::vector<DirectionMap<3, BlockNeighbor<3>>>{
            {{Direction<3>::lower_xi(), {0, {}}},
             {Direction<3>::upper_xi(), {0, {}}},
             {Direction<3>::lower_zeta(), {0, {}}},
             {Direction<3>::upper_zeta(), {0, {}}}}},
        std::vector<std::unordered_set<Direction<3>>>{
            {{Direction<3>::lower_eta()}, {Direction<3>::upper_eta()}}},
        std::make_tuple(
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                f_of_t_name,
                {initial_time,
                 std::array<DataVector, 3>{{{3, 0.0}, velocity, {3, 0.0}}},
                 initial_expiration_times[f_of_t_name]}}),
        make_vector_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            Translation3D{f_of_t_name}),
        {}, initial_expiration_times);
  }
  {
    INFO("Brick factory time dependent");
    const auto domain_creator = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<3>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithBoundaryConditions<3, domain::creators::Brick>>(
        "Brick:\n"
        "  LowerBound: [0,0,0]\n"
        "  UpperBound: [1,2,3]\n"
        "  InitialGridPoints: [3,4,3]\n"
        "  InitialRefinement: [2,3,2]\n"
        "  TimeDependence:\n"
        "    UniformTranslation:\n"
        "      InitialTime: 1.0\n"
        "      Velocity: [2.3, -0.3, 0.5]\n" +
        boundary_conditions);
    const auto* brick_creator =
        dynamic_cast<const creators::Brick*>(domain_creator.get());
    const double initial_time = 1.0;
    const DataVector velocity{{2.3, -0.3, 0.5}};
    // This name must match the hard coded one in UniformTranslation
    const std::string f_of_t_name = "Translation";
    std::unordered_map<std::string, double> initial_expiration_times{};
    initial_expiration_times[f_of_t_name] = 10.0;
    // without expiration times
    test_brick_construction(
        *brick_creator, {{0., 0., 0.}}, {{1., 2., 3.}}, {{{3, 4, 3}}},
        {{{2, 3, 2}}}, {{}},
        std::vector<std::unordered_set<Direction<3>>>{
            {{Direction<3>::lower_xi()},
             {Direction<3>::upper_xi()},
             {Direction<3>::lower_eta()},
             {Direction<3>::upper_eta()},
             {Direction<3>::lower_zeta()},
             {Direction<3>::upper_zeta()}}},
        std::make_tuple(
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                f_of_t_name,
                {initial_time,
                 std::array<DataVector, 3>{{{3, 0.0}, velocity, {3, 0.0}}},
                 std::numeric_limits<double>::infinity()}}),
        make_vector_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            Translation3D{f_of_t_name}),
        true);
    // with expiration times
    test_brick_construction(
        *brick_creator, {{0., 0., 0.}}, {{1., 2., 3.}}, {{{3, 4, 3}}},
        {{{2, 3, 2}}}, {{}},
        std::vector<std::unordered_set<Direction<3>>>{
            {{Direction<3>::lower_xi()},
             {Direction<3>::upper_xi()},
             {Direction<3>::lower_eta()},
             {Direction<3>::upper_eta()},
             {Direction<3>::lower_zeta()},
             {Direction<3>::upper_zeta()}}},
        std::make_tuple(
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                f_of_t_name,
                {initial_time,
                 std::array<DataVector, 3>{{{3, 0.0}, velocity, {3, 0.0}}},
                 initial_expiration_times[f_of_t_name]}}),
        make_vector_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            Translation3D{f_of_t_name}),
        true, initial_expiration_times);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.Brick", "[Domain][Unit]") {
  test_brick();
  test_brick_factory();
}
}  // namespace domain
