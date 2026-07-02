// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <unordered_set>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/CartoonCylinder.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Utilities/MakeVector.hpp"

namespace domain {
namespace {
using Affine = CoordinateMaps::Affine;
using Identity1D = CoordinateMaps::Identity<1>;
using cartoon_cylinder_map =
    CoordinateMaps::ProductOf3Maps<Affine, Affine, Identity1D>;
using Translation3D = CoordinateMaps::TimeDependent::Translation<3>;

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_cartoon_boundary_condition() {
  return std::make_unique<TestHelpers::domain::BoundaryConditions::
                              TestCartoonBoundaryCondition<3>>();
}

template <typename... FuncsOfTime>
void test_cylinder_construction(
    const creators::CartoonCylinder& cylinder,
    const std::array<double, 2>& lower_bounds,
    const std::array<double, 2>& upper_bounds,
    const std::vector<std::array<size_t, 3>>& expected_extents,
    const std::vector<std::array<size_t, 3>>& expected_refinement_level,
    const std::vector<DirectionMap<3, BlockNeighbors<3>>>&
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
      cylinder, expect_boundary_conditions);
  CHECK(cylinder.grid_anchors().empty());

  CHECK(cylinder.block_names() == std::vector<std::string>{"Block0"});
  const auto block_groups = cylinder.block_groups();
  CHECK(block_groups.contains("CartoonCylinder"));
  CHECK(block_groups.at("CartoonCylinder") ==
        std::unordered_set<std::string>{"Block0"});

  CHECK(cylinder.initial_extents() == expected_extents);
  CHECK(cylinder.initial_refinement_levels() == expected_refinement_level);

  if (lower_bounds[0] == 0.0) {
    CHECK(domain.blocks().front().topologies() ==
          domain::topologies::cartoon_cylinder_inner);
  } else {
    CHECK(domain.blocks().front().topologies() ==
          domain::topologies::cartoon_cylinder);
  }

  test_domain_construction(
      domain, expected_block_neighbors, expected_external_boundaries,
      make_vector(
          make_coordinate_map_base<
              Frame::BlockLogical,
              tmpl::conditional_t<sizeof...(FuncsOfTime) == 0, Frame::Inertial,
                                  Frame::Grid>>(cartoon_cylinder_map{
              Affine{-1., 1., lower_bounds[0], upper_bounds[0]},
              Affine{-1., 1., lower_bounds[1], upper_bounds[1]},
              Identity1D{}})),
      10.0, cylinder.functions_of_time(), expected_grid_to_inertial_maps);
  TestHelpers::domain::creators::test_functions_of_time(
      cylinder, expected_functions_of_time, initial_expiration_times);
}

void test_cylinder() {
  {
    INFO("CartoonCylinder");
    const std::vector<std::array<size_t, 2>> grid_points_creator{{{4, 6}}};
    const std::vector<std::array<size_t, 2>> refinement_level_creator{{{3, 2}}};
    const std::vector<std::array<size_t, 3>> grid_points{{{4, 6, 1}}};
    const std::vector<std::array<size_t, 3>> refinement_level{{{3, 2, 0}}};
    const std::array<double, 2> lower_bounds{{0.0, -3.0}};
    const std::array<double, 2> lower_bounds_no_zernike{{0.2, -3.0}};
    const std::array<double, 2> upper_bounds{{0.8, 5.0}};
    const OrientationMap<3> aligned_orientation =
        OrientationMap<3>::create_aligned();
    const TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>
        test_bc{Direction<3>::lower_xi(), 0};
    const TestHelpers::domain::BoundaryConditions::
        TestPeriodicBoundaryCondition<3>
            periodic_bc{};
    const TestHelpers::domain::BoundaryConditions::TestNoneBoundaryCondition<3>
        none_bc{};
    using bc_type = std::array<
        std::array<
            std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>, 2>,
        2>;

    {
      INFO("CartoonCylinder, non-periodic with Zernike");
      std::vector<DirectionMap<
          3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
          expected_boundary_conditions{1};
      for (const auto& direction : Direction<3>::all_directions()) {
        expected_boundary_conditions[0][direction] = test_bc.get_clone();
      }
      bc_type bc_array{{{{test_bc.get_clone(), test_bc.get_clone()}},
                        {{test_bc.get_clone(), test_bc.get_clone()}}}};
      test_cylinder_construction(
          creators::CartoonCylinder{lower_bounds,
                                    upper_bounds,
                                    refinement_level_creator[0],
                                    grid_points_creator[0],
                                    {},
                                    nullptr,
                                    std::move(bc_array),
                                    create_cartoon_boundary_condition()},
          lower_bounds, upper_bounds, grid_points, refinement_level,
          std::vector<DirectionMap<3, BlockNeighbors<3>>>{{}},
          std::vector<std::unordered_set<Direction<3>>>{
              {{Direction<3>::lower_xi()},
               {Direction<3>::upper_xi()},
               {Direction<3>::lower_eta()},
               {Direction<3>::upper_eta()}}},
          {}, {}, true);
    }
    {
      INFO("CartoonCylinder, non-periodic without Zernike");
      std::vector<DirectionMap<
          3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
          expected_boundary_conditions{1};
      for (const auto& direction : Direction<3>::all_directions()) {
        expected_boundary_conditions[0][direction] = test_bc.get_clone();
      }
      bc_type bc_array{{{{test_bc.get_clone(), test_bc.get_clone()}},
                        {{test_bc.get_clone(), test_bc.get_clone()}}}};
      test_cylinder_construction(
          creators::CartoonCylinder{lower_bounds_no_zernike,
                                    upper_bounds,
                                    refinement_level_creator[0],
                                    grid_points_creator[0],
                                    {},
                                    nullptr,
                                    std::move(bc_array),
                                    create_cartoon_boundary_condition()},
          lower_bounds_no_zernike, upper_bounds, grid_points, refinement_level,
          std::vector<DirectionMap<3, BlockNeighbors<3>>>{{}},
          std::vector<std::unordered_set<Direction<3>>>{
              {{Direction<3>::lower_xi()},
               {Direction<3>::upper_xi()},
               {Direction<3>::lower_eta()},
               {Direction<3>::upper_eta()}}},
          {}, {}, true);
    }
    {
      INFO("CartoonCylinder, periodic in y");
      bc_type bc_array{{{{test_bc.get_clone(), test_bc.get_clone()}},
                        {{periodic_bc.get_clone(), periodic_bc.get_clone()}}}};
      test_cylinder_construction(
          creators::CartoonCylinder{lower_bounds,
                                    upper_bounds,
                                    refinement_level_creator[0],
                                    grid_points_creator[0],
                                    {},
                                    nullptr,
                                    std::move(bc_array),
                                    create_cartoon_boundary_condition()},
          lower_bounds, upper_bounds, grid_points, refinement_level,
          std::vector<DirectionMap<3, BlockNeighbors<3>>>{
              {{Direction<3>::lower_eta(), {0, aligned_orientation}},
               {Direction<3>::upper_eta(), {0, aligned_orientation}}}},
          std::vector<std::unordered_set<Direction<3>>>{
              {{Direction<3>::lower_xi()}, {Direction<3>::upper_xi()}}},
          {}, {}, true);
    }
  }
  {
    const TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>
        test_bc{Direction<3>::lower_xi(), 0};
    const TestHelpers::domain::BoundaryConditions::
        TestPeriodicBoundaryCondition<3>
            periodic_bc{};
    const TestHelpers::domain::BoundaryConditions::TestNoneBoundaryCondition<3>
        none_bc{};
    // to avoid clang-tidy complaining about memory leak
    using bc_type = std::array<
        std::array<
            std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>, 2>,
        2>;
    {
      bc_type none_bc_arr{{{{none_bc.get_clone(), none_bc.get_clone()}},
                           {{test_bc.get_clone(), test_bc.get_clone()}}}};
      std::array<size_t, 2> grid_points_creator{{4, 6}};
      std::array<size_t, 2> refinement_level_creator{{3, 2}};
      std::array<double, 2> lower_bounds{{0.0, -3.0}};
      std::array<double, 2> upper_bounds{{0.8, 5.0}};
      CHECK_THROWS_WITH(
          creators::CartoonCylinder(
              std::move(lower_bounds), std::move(upper_bounds),
              std::move(refinement_level_creator),
              std::move(grid_points_creator), {}, nullptr,
              std::move(none_bc_arr), create_cartoon_boundary_condition(),
              Options::Context{false, {}, 1, 1}),
          Catch::Matchers::ContainsSubstring(
              "None boundary condition is not supported. If you would like an "
              "outflow-type boundary condition, you must use that."));
    }
    {
      bc_type periodic_bc_arr{
          {{{periodic_bc.get_clone(), periodic_bc.get_clone()}},
           {{test_bc.get_clone(), test_bc.get_clone()}}}};
      std::array<size_t, 2> grid_points_creator{{4, 6}};
      std::array<size_t, 2> refinement_level_creator{{3, 2}};
      std::array<double, 2> lower_bounds{{0.0, -3.0}};
      std::array<double, 2> upper_bounds{{0.8, 5.0}};
      CHECK_THROWS_WITH(
          creators::CartoonCylinder(
              std::move(lower_bounds), std::move(upper_bounds),
              std::move(refinement_level_creator),
              std::move(grid_points_creator), {}, nullptr,
              std::move(periodic_bc_arr), create_cartoon_boundary_condition(),
              Options::Context{false, {}, 1, 1}),
          Catch::Matchers::ContainsSubstring(
              "Cannot have periodic boundary conditions in the x dimension."));
    }
    {
      bc_type test_bc_arr{{{{test_bc.get_clone(), test_bc.get_clone()}},
                           {{test_bc.get_clone(), test_bc.get_clone()}}}};
      std::array<size_t, 2> grid_points_creator{{4, 6}};
      std::array<size_t, 2> refinement_level_creator{{3, 2}};
      std::array<double, 2> invalid_lower_bounds{{-0.3, 1.0}};
      std::array<double, 2> upper_bounds{{0.8, 5.0}};
      CHECK_THROWS_WITH(
          creators::CartoonCylinder(
              std::move(invalid_lower_bounds), std::move(upper_bounds),
              std::move(refinement_level_creator),
              std::move(grid_points_creator), {}, nullptr,
              std::move(test_bc_arr), create_cartoon_boundary_condition(),
              Options::Context{false, {}, 1, 1}),
          Catch::Matchers::ContainsSubstring(
              "The lower bound for the x dimension must be >= 0"));
    }
    {
      bc_type cartoon_bc_arr{
          {{{test_bc.get_clone(), test_bc.get_clone()}},
           {{create_cartoon_boundary_condition(), test_bc.get_clone()}}}};
      std::array<size_t, 2> grid_points_creator{{4, 6}};
      std::array<size_t, 2> refinement_level_creator{{3, 2}};
      std::array<double, 2> lower_bounds{{0.0, 1.0}};
      std::array<double, 2> upper_bounds{{0.8, 5.0}};
      CHECK_THROWS_WITH(
          creators::CartoonCylinder(
              std::move(lower_bounds), std::move(upper_bounds),
              std::move(refinement_level_creator),
              std::move(grid_points_creator), {}, nullptr,
              std::move(cartoon_bc_arr), create_cartoon_boundary_condition(),
              Options::Context{false, {}, 1, 1}),
          Catch::Matchers::ContainsSubstring(
              "Cartoon boundary conditions should not be specified as "));
    }
  }
}  // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)

void test_cylinder_factory() {
  // For non-periodic domains:
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      expected_boundary_conditions{1};
  for (const auto& direction : Direction<3>::all_directions()) {
    expected_boundary_conditions[0][direction] = std::make_unique<
        TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
        Direction<3>::lower_xi(), 0);
  }
  const std::vector<std::unordered_set<Direction<3>>>
      expected_external_boundaries{
          {Direction<3>::lower_xi(), Direction<3>::upper_xi(),
           Direction<3>::lower_eta(), Direction<3>::upper_eta(),
           Direction<3>::lower_zeta(), Direction<3>::upper_zeta()}};

  // for periodic domains:
  const std::vector<DirectionMap<3, BlockNeighbors<3>>> expected_neighbors{
      {{Direction<3>::lower_eta(), {0, OrientationMap<3>::create_aligned()}},
       {Direction<3>::upper_eta(), {0, OrientationMap<3>::create_aligned()}}}};

  {
    INFO("CartoonCylinder factory time independent");
    const auto domain_creator = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<3>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithBoundaryConditionsCartoon<
                3, domain::creators::CartoonCylinder>>(
        "CartoonCylinder:\n"
        "  LowerBounds: [0,0]\n"
        "  UpperBounds: [1,2]\n"
        "  Distributions: [Linear, Linear]\n"
        "  InitialGridPoints: [3,4]\n"
        "  InitialRefinement: [2,3]\n"
        "  TimeDependence: None\n"
        "  BoundaryConditions:\n"
        "    - TestBoundaryCondition:\n"
        "        Direction: lower-xi\n"
        "        BlockId: 0\n"
        "    - TestBoundaryCondition:\n"
        "        Direction: lower-xi\n"
        "        BlockId: 0\n");
    const auto* cylinder_creator =
        dynamic_cast<const creators::CartoonCylinder*>(domain_creator.get());
    test_cylinder_construction(
        *cylinder_creator, {{0., 0.}}, {{1., 2.}}, {{{3, 4, 1}}}, {{{2, 3, 0}}},
        {{}},
        std::vector<std::unordered_set<Direction<3>>>{
            {{Direction<3>::lower_xi(), Direction<3>::upper_xi(),
              Direction<3>::lower_eta(), Direction<3>::upper_eta()}}},
        {}, {}, true);
  }
  {
    INFO("CartoonCylinder factory time dependent, with boundary conditions");
    const auto domain_creator = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<3>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithBoundaryConditionsCartoon<
                3, domain::creators::CartoonCylinder>>(
        "CartoonCylinder:\n"
        "  LowerBounds: [0,0]\n"
        "  UpperBounds: [1,2]\n"
        "  Distributions: [Linear, Linear]\n"
        "  InitialGridPoints: [3,4]\n"
        "  InitialRefinement: [2,3]\n"
        "  TimeDependence:\n"
        "    UniformTranslation:\n"
        "      InitialTime: 1.0\n"
        "      Velocity: [2.3, -0.3, 0.5]\n"
        "  BoundaryConditions:\n"
        "    - TestBoundaryCondition:\n"
        "        Direction: lower-xi\n"
        "        BlockId: 0\n"
        "    - TestBoundaryCondition:\n"
        "        Direction: lower-xi\n"
        "        BlockId: 0\n");
    const auto* cylinder_creator =
        dynamic_cast<const creators::CartoonCylinder*>(domain_creator.get());
    const double initial_time = 1.0;
    const DataVector velocity{{2.3, -0.3, 0.5}};
    // This name must match the hard coded one in UniformTranslation
    const std::string f_of_t_name = "Translation";
    std::unordered_map<std::string, double> initial_expiration_times{};
    initial_expiration_times[f_of_t_name] = 10.0;
    // without expiration times
    test_cylinder_construction(
        *cylinder_creator, {{0., 0.}}, {{1., 2.}}, {{{3, 4, 1}}}, {{{2, 3, 0}}},
        {{}},
        std::vector<std::unordered_set<Direction<3>>>{
            {{Direction<3>::lower_xi(), Direction<3>::upper_xi(),
              Direction<3>::lower_eta(), Direction<3>::upper_eta()}}},
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
    test_cylinder_construction(
        *cylinder_creator, {{0., 0.}}, {{1., 2.}}, {{{3, 4, 1}}}, {{{2, 3, 0}}},
        {{}},
        std::vector<std::unordered_set<Direction<3>>>{
            {{Direction<3>::lower_xi(), Direction<3>::upper_xi(),
              Direction<3>::lower_eta(), Direction<3>::upper_eta()}}},
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
  CHECK_THROWS_WITH(
      (TestHelpers::test_option_tag<
          domain::OptionTags::DomainCreator<3>,
          TestHelpers::domain::BoundaryConditions::
              MetavariablesWithBoundaryConditions<
                  3, domain::creators::CartoonCylinder>>(
          "CartoonCylinder:\n"
          "  LowerBounds: [0,0]\n"
          "  UpperBounds: [1,2]\n"
          "  Distributions: [Linear, Linear]\n"
          "  InitialGridPoints: [3,4]\n"
          "  InitialRefinement: [2,3]\n"
          "  TimeDependence: None\n"
          "  BoundaryConditions:\n"
          "    - TestBoundaryCondition:\n"
          "        Direction: lower-xi\n"
          "        BlockId: 0\n"
          "    - TestBoundaryCondition:\n"
          "        Direction: lower-xi\n"
          "        BlockId: 0\n")),
      Catch::Matchers::ContainsSubstring(
          "CartoonCylinder should only be used with systems that have a "
          "cartoon-style boundary condition"));
}  // namespace domain
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.CartoonCylinder", "[Domain][Unit]") {
  test_cylinder();
  test_cylinder_factory();
}
}  // namespace domain
