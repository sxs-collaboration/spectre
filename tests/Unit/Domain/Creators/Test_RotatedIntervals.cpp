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
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Creators/RotatedIntervals.hpp"
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
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/MakeVector.hpp"

namespace domain {
namespace {
template <typename... FuncsOfTime>
void test_rotated_intervals_construction(
    const creators::RotatedIntervals& rotated_intervals,
    const std::array<double, 1>& lower_bound,
    const std::array<double, 1>& midpoint,
    const std::array<double, 1>& upper_bound,
    const std::vector<std::array<size_t, 1>>& expected_extents,
    const std::vector<std::array<size_t, 1>>& expected_refinement_level,
    const std::vector<DirectionMap<1, BlockNeighbor<1>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<Direction<1>>>&
        expected_external_boundaries,
    const std::tuple<std::pair<std::string, FuncsOfTime>...>&
        expected_functions_of_time,
    const std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 1>>>&
        expected_grid_to_inertial_maps,
    const bool expect_boundary_conditions = false,
    const bool is_periodic = false,
    const std::unordered_map<std::string, double>& initial_expiration_times =
        {}) {
  const std::vector<double> times{1.};
  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      rotated_intervals, expect_boundary_conditions, is_periodic, times);

  CHECK(rotated_intervals.initial_extents() == expected_extents);
  CHECK(rotated_intervals.initial_refinement_levels() ==
        expected_refinement_level);

  test_domain_construction(
      domain, expected_block_neighbors, expected_external_boundaries,
      make_vector(
          make_coordinate_map_base<
              Frame::BlockLogical,
              tmpl::conditional_t<sizeof...(FuncsOfTime) == 0, Frame::Inertial,
                                  Frame::Grid>>(
              CoordinateMaps::Affine{-1., 1., lower_bound[0], midpoint[0]}),
          make_coordinate_map_base<
              Frame::BlockLogical,
              tmpl::conditional_t<sizeof...(FuncsOfTime) == 0, Frame::Inertial,
                                  Frame::Grid>>(
              CoordinateMaps::DiscreteRotation<1>{OrientationMap<1>{
                  std::array<Direction<1>, 1>{{Direction<1>::lower_xi()}}}},
              CoordinateMaps::Affine{-1., 1., midpoint[0], upper_bound[0]})),
      10.0, rotated_intervals.functions_of_time(),
      expected_grid_to_inertial_maps);
  TestHelpers::domain::creators::test_functions_of_time(
      rotated_intervals, expected_functions_of_time, initial_expiration_times);
}

void test_rotated_intervals() {
  INFO("Rotated intervals");
  const std::vector<std::array<size_t, 1>> grid_points{{{4}}, {{2}}};
  const std::vector<std::array<size_t, 1>> refinement_level{{{0}}, {{0}}};
  const std::array<double, 1> lower_bound{{-1.2}};
  const std::array<double, 1> midpoint{{-0.6}};
  const std::array<double, 1> upper_bound{{0.8}};
  const OrientationMap<1> flipped{
      std::array<Direction<1>, 1>{{Direction<1>::lower_xi()}}};

  const auto check_nonperiodic =
      [&lower_bound, &midpoint, &upper_bound, &grid_points, &refinement_level,
       &flipped](const creators::RotatedIntervals& rotated_intervals,
                 const bool expect_bcs) {
        test_rotated_intervals_construction(
            rotated_intervals, lower_bound, midpoint, upper_bound, grid_points,
            refinement_level,
            std::vector<DirectionMap<1, BlockNeighbor<1>>>{
                {{Direction<1>::upper_xi(), {1, flipped}}},
                {{Direction<1>::upper_xi(), {0, flipped}}}},
            std::vector<std::unordered_set<Direction<1>>>{
                {Direction<1>::lower_xi()}, {Direction<1>::lower_xi()}},
            std::tuple<>{}, {}, expect_bcs);
      };

  const auto check_periodic =
      [&lower_bound, &midpoint, &upper_bound, &grid_points, &refinement_level,
       &flipped](const creators::RotatedIntervals& periodic_rotated_intervals,
                 const bool expect_bcs) {
        test_rotated_intervals_construction(
            periodic_rotated_intervals, lower_bound, midpoint, upper_bound,
            grid_points, refinement_level,
            std::vector<DirectionMap<1, BlockNeighbor<1>>>{
                {{Direction<1>::lower_xi(), {1, flipped}},
                 {Direction<1>::upper_xi(), {1, flipped}}},
                {{Direction<1>::lower_xi(), {0, flipped}},
                 {Direction<1>::upper_xi(), {0, flipped}}}},
            std::vector<std::unordered_set<Direction<1>>>{{}, {}},
            std::tuple<>{}, {}, expect_bcs, true);
      };

  {
    INFO("Check non-periodic via array of bools");
    check_nonperiodic({lower_bound,
                       midpoint,
                       upper_bound,
                       refinement_level[0],
                       {{{{grid_points[0][0], grid_points[1][0]}}}},
                       std::array<bool, 1>{{false}},
                       nullptr},
                      false);
  }

  {
    INFO("Check periodic via array of bools");
    check_periodic({lower_bound,
                    midpoint,
                    upper_bound,
                    refinement_level[0],
                    {{{{grid_points[0][0], grid_points[1][0]}}}},
                    std::array<bool, 1>{{true}},
                    nullptr},
                   false);
  }

  // Test with boundary conditions
  const auto lower_bc = std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<1>>(
      Direction<1>::lower_xi(), 0);
  const auto upper_bc = std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<1>>(
      Direction<1>::upper_xi(), 1);
  {
    INFO("Check non-periodic with boundary conditions");
    check_nonperiodic({lower_bound,
                       midpoint,
                       upper_bound,
                       refinement_level[0],
                       {{{{grid_points[0][0], grid_points[1][0]}}}},
                       lower_bc->get_clone(),
                       upper_bc->get_clone(),
                       nullptr},
                      true);
  }

  const std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      periodic = std::make_unique<TestHelpers::domain::BoundaryConditions::
                                      TestPeriodicBoundaryCondition<1>>();
  {
    INFO("Check periodic via boundary conditions");
    check_periodic({lower_bound,
                    midpoint,
                    upper_bound,
                    refinement_level[0],
                    {{{{grid_points[0][0], grid_points[1][0]}}}},
                    periodic->get_clone(),
                    periodic->get_clone(),
                    nullptr},
                   true);
  }

  // Test parse error
  CHECK_THROWS_WITH(
      creators::RotatedIntervals(
          lower_bound, midpoint, upper_bound, refinement_level[0],
          {{{{grid_points[0][0], grid_points[1][0]}}}}, lower_bc->get_clone(),
          periodic->get_clone(), nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains("Both the upper and lower boundary condition "
                                "must be set to periodic if"));
  CHECK_THROWS_WITH(
      creators::RotatedIntervals(
          lower_bound, midpoint, upper_bound, refinement_level[0],
          {{{{grid_points[0][0], grid_points[1][0]}}}}, periodic->get_clone(),
          lower_bc->get_clone(), nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains("Both the upper and lower boundary condition "
                                "must be set to periodic if"));
  CHECK_THROWS_WITH(
      creators::RotatedIntervals(
          lower_bound, midpoint, upper_bound, refinement_level[0],
          {{{{grid_points[0][0], grid_points[1][0]}}}},
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          lower_bc->get_clone(), nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "None boundary condition is not supported. If you would like "
          "an outflow-type boundary condition, you must use that."));
  CHECK_THROWS_WITH(
      creators::RotatedIntervals(
          lower_bound, midpoint, upper_bound, refinement_level[0],
          {{{{grid_points[0][0], grid_points[1][0]}}}}, lower_bc->get_clone(),
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "None boundary condition is not supported. If you would like "
          "an outflow-type boundary condition, you must use that."));
}

void test_rotated_intervals_factory() {
  const OrientationMap<1> flipped{
      std::array<Direction<1>, 1>{{Direction<1>::lower_xi()}}};
  const auto lower_bc = std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<1>>(
      Direction<1>::lower_xi(), 0);
  const auto upper_bc = std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<1>>(
      Direction<1>::upper_xi(), 1);
  const std::vector<std::unordered_set<Direction<1>>>
      expected_external_boundaries{{Direction<1>::lower_xi()},
                                   {Direction<1>::lower_xi()}};
  {
    INFO("Rotated intervals factory time independent, no boundary condition");
    const auto domain_creator = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<1>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithoutBoundaryConditions<
                1, domain::creators::RotatedIntervals>>(
        "RotatedIntervals:\n"
        "  LowerBound: [0.0]\n"
        "  Midpoint:   [0.5]\n"
        "  UpperBound: [1.0]\n"
        "  IsPeriodicIn: [True]\n"
        "  InitialGridPoints: [[3,2]]\n"
        "  InitialRefinement: [2]\n"
        "  TimeDependence: None\n");
    const auto* rotated_intervals_creator =
        dynamic_cast<const creators::RotatedIntervals*>(domain_creator.get());
    test_rotated_intervals_construction(
        *rotated_intervals_creator, {{0.0}}, {{0.5}}, {{1.0}}, {{{3}}, {{2}}},
        {{{2}}, {{2}}},
        std::vector<DirectionMap<1, BlockNeighbor<1>>>{
            {{Direction<1>::lower_xi(), {1, flipped}},
             {Direction<1>::upper_xi(), {1, flipped}}},
            {{Direction<1>::lower_xi(), {0, flipped}},
             {Direction<1>::upper_xi(), {0, flipped}}}},
        std::vector<std::unordered_set<Direction<1>>>{{}, {}}, {}, {}, false,
        true);
  }
  {
    INFO("Rotated intervals factory time independent, with boundary condition");
    const auto domain_creator = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<1>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithBoundaryConditions<
                1, domain::creators::RotatedIntervals>>(
        "RotatedIntervals:\n"
        "  LowerBound: [0.0]\n"
        "  Midpoint:   [0.5]\n"
        "  UpperBound: [1.0]\n"
        "  InitialGridPoints: [[3,2]]\n"
        "  InitialRefinement: [2]\n"
        "  TimeDependence: None\n"
        "  BoundaryConditions:\n"
        "    LowerBoundary:\n"
        "      TestBoundaryCondition:\n"
        "        Direction: lower-xi\n"
        "        BlockId: 0\n"
        "    UpperBoundary:\n"
        "      TestBoundaryCondition:\n"
        "        Direction: lower-xi\n"
        "        BlockId: 1\n");
    const auto* rotated_intervals_creator =
        dynamic_cast<const creators::RotatedIntervals*>(domain_creator.get());
    test_rotated_intervals_construction(
        *rotated_intervals_creator, {{0.0}}, {{0.5}}, {{1.0}}, {{{3}}, {{2}}},
        {{{2}}, {{2}}},
        std::vector<DirectionMap<1, BlockNeighbor<1>>>{
            {{Direction<1>::upper_xi(), {1, flipped}}},
            {{Direction<1>::upper_xi(), {0, flipped}}}},
        expected_external_boundaries, {}, {}, true);
  }
  {
    INFO("Rotated intervals factory time dependent, no boundary condition");
    const auto domain_creator = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<1>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithoutBoundaryConditions<
                1, domain::creators::RotatedIntervals>>(
        "RotatedIntervals:\n"
        "  LowerBound: [0.0]\n"
        "  Midpoint:   [0.5]\n"
        "  UpperBound: [1.0]\n"
        "  IsPeriodicIn: [True]\n"
        "  InitialGridPoints: [[3,2]]\n"
        "  InitialRefinement: [2]\n"
        "  TimeDependence:\n"
        "    UniformTranslation:\n"
        "      InitialTime: 1.0\n"
        "      Velocity: [2.3]\n");
    const auto* rotated_intervals_creator =
        dynamic_cast<const creators::RotatedIntervals*>(domain_creator.get());
    const double initial_time = 1.0;
    const DataVector velocity{{2.3}};
    // This name must match the hard coded one in UniformTranslation
    const std::string f_of_t_name = "Translation";
    std::unordered_map<std::string, double> initial_expiration_times{};
    initial_expiration_times[f_of_t_name] = 10.0;
    // without expiration times
    test_rotated_intervals_construction(
        *rotated_intervals_creator, {{0.0}}, {{0.5}}, {{1.0}}, {{{3}}, {{2}}},
        {{{2}}, {{2}}},
        std::vector<DirectionMap<1, BlockNeighbor<1>>>{
            {{Direction<1>::lower_xi(), {1, flipped}},
             {Direction<1>::upper_xi(), {1, flipped}}},
            {{Direction<1>::lower_xi(), {0, flipped}},
             {Direction<1>::upper_xi(), {0, flipped}}}},
        std::vector<std::unordered_set<Direction<1>>>{{}, {}},
        std::make_tuple(
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                f_of_t_name,
                {initial_time,
                 std::array<DataVector, 3>{{{0.0}, velocity, {0.0}}},
                 std::numeric_limits<double>::infinity()}}),
        make_vector_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            CoordinateMaps::TimeDependent::Translation<1>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<1>{f_of_t_name}),
        false, true);
    // with expiration times
    test_rotated_intervals_construction(
        *rotated_intervals_creator, {{0.0}}, {{0.5}}, {{1.0}}, {{{3}}, {{2}}},
        {{{2}}, {{2}}},
        std::vector<DirectionMap<1, BlockNeighbor<1>>>{
            {{Direction<1>::lower_xi(), {1, flipped}},
             {Direction<1>::upper_xi(), {1, flipped}}},
            {{Direction<1>::lower_xi(), {0, flipped}},
             {Direction<1>::upper_xi(), {0, flipped}}}},
        std::vector<std::unordered_set<Direction<1>>>{{}, {}},
        std::make_tuple(
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                f_of_t_name,
                {initial_time,
                 std::array<DataVector, 3>{{{0.0}, velocity, {0.0}}},
                 initial_expiration_times[f_of_t_name]}}),
        make_vector_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            CoordinateMaps::TimeDependent::Translation<1>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<1>{f_of_t_name}),
        false, true, initial_expiration_times);
  }
  {
    INFO("Rotated intervals factory time dependent, with boundary condition");
    const auto domain_creator = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<1>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithBoundaryConditions<
                1, domain::creators::RotatedIntervals>>(
        "RotatedIntervals:\n"
        "  LowerBound: [0.0]\n"
        "  Midpoint:   [0.5]\n"
        "  UpperBound: [1.0]\n"
        "  InitialGridPoints: [[3,2]]\n"
        "  InitialRefinement: [2]\n"
        "  TimeDependence:\n"
        "    UniformTranslation:\n"
        "      InitialTime: 1.0\n"
        "      Velocity: [2.3]\n"
        "  BoundaryConditions:\n"
        "    LowerBoundary:\n"
        "      TestBoundaryCondition:\n"
        "        Direction: lower-xi\n"
        "        BlockId: 0\n"
        "    UpperBoundary:\n"
        "      TestBoundaryCondition:\n"
        "        Direction: lower-xi\n"
        "        BlockId: 1\n");
    const auto* rotated_intervals_creator =
        dynamic_cast<const creators::RotatedIntervals*>(domain_creator.get());
    const double initial_time = 1.0;
    const DataVector velocity{{2.3}};
    // This name must match the hard coded one in UniformTranslation
    const std::string f_of_t_name = "Translation";
    std::unordered_map<std::string, double> initial_expiration_times{};
    initial_expiration_times[f_of_t_name] = 10.0;
    // without expiration times
    test_rotated_intervals_construction(
        *rotated_intervals_creator, {{0.0}}, {{0.5}}, {{1.0}}, {{{3}}, {{2}}},
        {{{2}}, {{2}}},
        std::vector<DirectionMap<1, BlockNeighbor<1>>>{
            {{Direction<1>::upper_xi(), {1, flipped}}},
            {{Direction<1>::upper_xi(), {0, flipped}}}},
        expected_external_boundaries,
        std::make_tuple(
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                f_of_t_name,
                {initial_time,
                 std::array<DataVector, 3>{{{0.0}, velocity, {0.0}}},
                 std::numeric_limits<double>::infinity()}}),
        make_vector_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            CoordinateMaps::TimeDependent::Translation<1>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<1>{f_of_t_name}),
        true);
    // with expiration times
    test_rotated_intervals_construction(
        *rotated_intervals_creator, {{0.0}}, {{0.5}}, {{1.0}}, {{{3}}, {{2}}},
        {{{2}}, {{2}}},
        std::vector<DirectionMap<1, BlockNeighbor<1>>>{
            {{Direction<1>::upper_xi(), {1, flipped}}},
            {{Direction<1>::upper_xi(), {0, flipped}}}},
        expected_external_boundaries,
        std::make_tuple(
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                f_of_t_name,
                {initial_time,
                 std::array<DataVector, 3>{{{0.0}, velocity, {0.0}}},
                 initial_expiration_times[f_of_t_name]}}),
        make_vector_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            CoordinateMaps::TimeDependent::Translation<1>{f_of_t_name},
            CoordinateMaps::TimeDependent::Translation<1>{f_of_t_name}),
        true, false, initial_expiration_times);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.RotatedIntervals", "[Domain][Unit]") {
  test_rotated_intervals();
  test_rotated_intervals_factory();
}
}  // namespace domain
