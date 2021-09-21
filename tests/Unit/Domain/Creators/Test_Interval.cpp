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
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/OptionTags.hpp"
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
void test_interval_construction(
    const creators::Interval& interval,
    const std::array<double, 1>& lower_bound,
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
    const std::vector<DirectionMap<
        1, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>&
        expected_boundary_conditions) {
  const auto domain = interval.create_domain();

  CHECK(interval.initial_extents() == expected_extents);
  CHECK(interval.initial_refinement_levels() == expected_refinement_level);

  test_domain_construction(
      domain, expected_block_neighbors, expected_external_boundaries,
      make_vector(make_coordinate_map_base<
                  Frame::BlockLogical,
                  tmpl::conditional_t<sizeof...(FuncsOfTime) == 0,
                                      Frame::Inertial, Frame::Grid>>(
          CoordinateMaps::Affine{-1., 1., lower_bound[0], upper_bound[0]})),
      10.0, interval.functions_of_time(), expected_grid_to_inertial_maps,
      expected_boundary_conditions);
  test_initial_domain(domain, interval.initial_refinement_levels());
  TestHelpers::domain::creators::test_functions_of_time(
      interval, expected_functions_of_time);

  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  test_serialization(domain);
}

void test_interval() {
  INFO("Interval");
  const std::vector<std::array<size_t, 1>> grid_points{{{4}}};
  const std::vector<std::array<size_t, 1>> refinement_level{{{3}}};
  const std::array<double, 1> lower_bound{{-1.2}};
  const std::array<double, 1> upper_bound{{0.8}};
  // default Orientation is aligned
  const OrientationMap<1> aligned_orientation{};

  {
    INFO("No boundary condition");
    const creators::Interval interval{lower_bound,
                                      upper_bound,
                                      refinement_level[0],
                                      grid_points[0],
                                      std::array<bool, 1>{{false}},
                                      nullptr};

    // This domain creator doesn't support block names and groups yet, so they
    // should be empty
    CHECK(interval.block_names().empty());
    CHECK(interval.block_groups().empty());

    test_interval_construction(
        interval, lower_bound, upper_bound, grid_points, refinement_level,
        std::vector<DirectionMap<1, BlockNeighbor<1>>>{{}},
        std::vector<std::unordered_set<Direction<1>>>{
            {{Direction<1>::lower_xi()}, {Direction<1>::upper_xi()}}},
        std::tuple<>{}, {}, {});

    const creators::Interval periodic_interval{lower_bound,
                                               upper_bound,
                                               refinement_level[0],
                                               grid_points[0],
                                               std::array<bool, 1>{{true}},
                                               nullptr};
    test_interval_construction(
        periodic_interval, lower_bound, upper_bound, grid_points,
        refinement_level,
        std::vector<DirectionMap<1, BlockNeighbor<1>>>{
            {{Direction<1>::lower_xi(), {0, aligned_orientation}},
             {Direction<1>::upper_xi(), {0, aligned_orientation}}}},
        std::vector<std::unordered_set<Direction<1>>>{{}}, std::tuple<>{}, {},
        {});
  }
  {
    INFO("With boundary condition");
    std::vector<DirectionMap<
        1, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
        expected_boundary_conditions{1};
    expected_boundary_conditions[0][Direction<1>::lower_xi()] =
        std::make_unique<
            TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<1>>(
            Direction<1>::lower_xi(), 0);
    expected_boundary_conditions[0][Direction<1>::upper_xi()] =
        std::make_unique<
            TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<1>>(
            Direction<1>::upper_xi(), 0);
    const creators::Interval interval{
        lower_bound,
        upper_bound,
        refinement_level[0],
        grid_points[0],
        expected_boundary_conditions[0][Direction<1>::lower_xi()]->get_clone(),
        expected_boundary_conditions[0][Direction<1>::upper_xi()]->get_clone(),
        nullptr};
    test_interval_construction(
        interval, lower_bound, upper_bound, grid_points, refinement_level,
        std::vector<DirectionMap<1, BlockNeighbor<1>>>{{}},
        std::vector<std::unordered_set<Direction<1>>>{
            {{Direction<1>::lower_xi()}, {Direction<1>::upper_xi()}}},
        std::tuple<>{}, {}, expected_boundary_conditions);

    const std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        periodic = std::make_unique<TestHelpers::domain::BoundaryConditions::
                                        TestPeriodicBoundaryCondition<1>>();
    const creators::Interval periodic_interval{
        lower_bound,    upper_bound,           refinement_level[0],
        grid_points[0], periodic->get_clone(), periodic->get_clone(),
        nullptr};
    test_interval_construction(
        periodic_interval, lower_bound, upper_bound, grid_points,
        refinement_level,
        std::vector<DirectionMap<1, BlockNeighbor<1>>>{
            {{Direction<1>::lower_xi(), {0, aligned_orientation}},
             {Direction<1>::upper_xi(), {0, aligned_orientation}}}},
        std::vector<std::unordered_set<Direction<1>>>{{}}, std::tuple<>{}, {},
        {});

    // Test parse error
    CHECK_THROWS_WITH(
        creators::Interval(
            lower_bound, upper_bound, refinement_level[0], grid_points[0],
            expected_boundary_conditions[0][Direction<1>::lower_xi()]
                ->get_clone(),
            periodic->get_clone(), nullptr, Options::Context{false, {}, 1, 1}),
        Catch::Matchers::Contains("Both the upper and lower boundary condition "
                                  "must be set to periodic if"));
    CHECK_THROWS_WITH(
        creators::Interval(
            lower_bound, upper_bound, refinement_level[0], grid_points[0],
            periodic->get_clone(),
            expected_boundary_conditions[0][Direction<1>::lower_xi()]
                ->get_clone(),
            nullptr, Options::Context{false, {}, 1, 1}),
        Catch::Matchers::Contains("Both the upper and lower boundary condition "
                                  "must be set to periodic if"));
    CHECK_THROWS_WITH(
        creators::Interval(
            lower_bound, upper_bound, refinement_level[0], grid_points[0],
            expected_boundary_conditions[0][Direction<1>::lower_xi()]
                ->get_clone(),
            std::make_unique<TestHelpers::domain::BoundaryConditions::
                                 TestNoneBoundaryCondition<3>>(),
            nullptr, Options::Context{false, {}, 1, 1}),
        Catch::Matchers::Contains(
            "None boundary condition is not supported. If you would like an "
            "outflow boundary condition, you must use that."));
    CHECK_THROWS_WITH(
        creators::Interval(
            lower_bound, upper_bound, refinement_level[0], grid_points[0],
            std::make_unique<TestHelpers::domain::BoundaryConditions::
                                 TestNoneBoundaryCondition<3>>(),
            expected_boundary_conditions[0][Direction<1>::lower_xi()]
                ->get_clone(),
            nullptr, Options::Context{false, {}, 1, 1}),
        Catch::Matchers::Contains(
            "None boundary condition is not supported. If you would like an "
            "outflow boundary condition, you must use that."));
  }
}

void test_interval_factory() {
  // For non-periodic domains:
  std::vector<DirectionMap<
      1, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      expected_boundary_conditions{1};
  expected_boundary_conditions[0][Direction<1>::lower_xi()] = std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<1>>(
      Direction<1>::lower_xi(), 0);
  expected_boundary_conditions[0][Direction<1>::upper_xi()] = std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<1>>(
      Direction<1>::upper_xi(), 0);
  const std::vector<std::unordered_set<Direction<1>>>
      expected_external_boundaries{
          {Direction<1>::lower_xi(), Direction<1>::upper_xi()}};

  // for periodic domains:
  const std::vector<DirectionMap<1, BlockNeighbor<1>>> expected_neighbors{
      {{Direction<1>::lower_xi(), {0, {}}},
       {Direction<1>::upper_xi(), {0, {}}}}};
  {
    INFO("Interval factory time independent, no boundary condition");
    const auto domain_creator = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<1>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithoutBoundaryConditions<
                1, domain::creators::Interval>>(
        "Interval:\n"
        "  LowerBound: [0]\n"
        "  UpperBound: [1]\n"
        "  IsPeriodicIn: [True]\n"
        "  InitialGridPoints: [3]\n"
        "  InitialRefinement: [2]\n"
        "  TimeDependence: None\n");
    const auto* interval_creator =
        dynamic_cast<const creators::Interval*>(domain_creator.get());
    test_interval_construction(*interval_creator, {{0.}}, {{1.}}, {{{3}}},
                               {{{2}}}, expected_neighbors, {{}}, {}, {}, {});
  }
  {
    INFO("Interval factory time independent, with boundary condition");
    const auto domain_creator = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<1>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithBoundaryConditions<1, domain::creators::Interval>>(
        "Interval:\n"
        "  LowerBound: [0]\n"
        "  UpperBound: [1]\n"
        "  InitialGridPoints: [3]\n"
        "  InitialRefinement: [2]\n"
        "  TimeDependence: None\n"
        "  BoundaryConditions:\n"
        "    LowerBoundary:\n"
        "      TestBoundaryCondition:\n"
        "        Direction: lower-xi\n"
        "        BlockId: 0\n"
        "    UpperBoundary:\n"
        "      TestBoundaryCondition:\n"
        "        Direction: upper-xi\n"
        "        BlockId: 0\n");
    const auto* interval_creator =
        dynamic_cast<const creators::Interval*>(domain_creator.get());
    test_interval_construction(*interval_creator, {{0.}}, {{1.}}, {{{3}}},
                               {{{2}}}, {{}}, expected_external_boundaries, {},
                               {}, expected_boundary_conditions);
  }
  {
    INFO("Interval factory time dependent, no boundary condition");
    const auto domain_creator = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<1>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithoutBoundaryConditions<
                1, domain::creators::Interval>>(
        "Interval:\n"
        "  LowerBound: [0]\n"
        "  UpperBound: [1]\n"
        "  IsPeriodicIn: [True]\n"
        "  InitialGridPoints: [3]\n"
        "  InitialRefinement: [2]\n"
        "  TimeDependence:\n"
        "    UniformTranslation:\n"
        "      InitialTime: 1.0\n"
        "      InitialExpirationDeltaT: 9.0\n"
        "      Velocity: [2.3]\n"
        "      FunctionOfTimeName: TranslationX");
    const auto* interval_creator =
        dynamic_cast<const creators::Interval*>(domain_creator.get());
    test_interval_construction(
        *interval_creator, {{0.}}, {{1.}}, {{{3}}}, {{{2}}}, expected_neighbors,
        std::vector<std::unordered_set<Direction<1>>>{{}},
        std::make_tuple(
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                "TranslationX",
                {1.0, std::array<DataVector, 3>{{{0.0}, {2.3}, {0.0}}}, 10.0}}),
        make_vector_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            CoordinateMaps::TimeDependent::Translation<1>{"TranslationX"}),
        {});
  }
  {
    INFO("Interval factory time dependent, with boundary condition");
    const auto domain_creator = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<1>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithBoundaryConditions<1, domain::creators::Interval>>(
        "Interval:\n"
        "  LowerBound: [0]\n"
        "  UpperBound: [1]\n"
        "  InitialGridPoints: [3]\n"
        "  InitialRefinement: [2]\n"
        "  TimeDependence:\n"
        "    UniformTranslation:\n"
        "      InitialTime: 1.0\n"
        "      InitialExpirationDeltaT: 9.0\n"
        "      Velocity: [2.3]\n"
        "      FunctionOfTimeName: TranslationX\n"
        "  BoundaryConditions:\n"
        "    LowerBoundary:\n"
        "      TestBoundaryCondition:\n"
        "        Direction: lower-xi\n"
        "        BlockId: 0\n"
        "    UpperBoundary:\n"
        "      TestBoundaryCondition:\n"
        "        Direction: upper-xi\n"
        "        BlockId: 0\n");
    const auto* interval_creator =
        dynamic_cast<const creators::Interval*>(domain_creator.get());
    test_interval_construction(
        *interval_creator, {{0.}}, {{1.}}, {{{3}}}, {{{2}}}, {{}},
        expected_external_boundaries,
        std::make_tuple(
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                "TranslationX",
                {1.0, std::array<DataVector, 3>{{{0.0}, {2.3}, {0.0}}}, 10.0}}),
        make_vector_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            CoordinateMaps::TimeDependent::Translation<1>{"TranslationX"}),
        expected_boundary_conditions);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.Interval.Factory", "[Domain][Unit]") {
  test_interval();
  test_interval_factory();
}
}  // namespace domain
