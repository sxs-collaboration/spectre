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
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/PolarToCartesian.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/AngularDisk.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Utilities/MakeVector.hpp"

namespace domain {
namespace {

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_boundary_condition() {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<2>>(
      Direction<2>::lower_xi(), 0);
}

template <typename... FuncsOfTime>
void test_angular_disk_construction(
    const creators::AngularDisk& angular_disk, const double outer_radius,
    const std::vector<double>& radial_partitioning,
    const std::vector<std::array<size_t, 2>>& expected_extents,
    const std::vector<std::array<size_t, 2>>& expected_refinement_level,
    const std::vector<DirectionMap<2, BlockNeighbors<2>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<Direction<2>>>&
        expected_external_boundaries,
    const std::tuple<std::pair<std::string, FuncsOfTime>...>&
        expected_functions_of_time = {},
    const std::vector<std::unique_ptr<domain::CoordinateMapBase<
        Frame::Grid, Frame::Inertial, 2>>>& expected_grid_to_inertial_maps = {},
    const bool expect_boundary_conditions = false,
    const std::unordered_map<std::string, double>& initial_expiration_times =
        {}) {
  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      angular_disk, expect_boundary_conditions);
  CHECK(angular_disk.grid_anchors().empty());

  // Verify block names
  const size_t num_blocks = 1 + radial_partitioning.size();
  std::vector<std::string> expected_block_names;
  expected_block_names.emplace_back("InnerDisk");
  for (size_t i = 1; i < num_blocks; ++i) {
    expected_block_names.emplace_back("Shell" + std::to_string(i - 1));
  }
  CHECK(angular_disk.block_names() == expected_block_names);

  // Verify block groups
  const auto block_groups = angular_disk.block_groups();
  CHECK(block_groups.contains("InnerDisk"));
  CHECK(block_groups.at("InnerDisk") ==
        std::unordered_set<std::string>{"InnerDisk"});

  if (num_blocks > 1) {
    CHECK(block_groups.contains("Shells"));
    std::unordered_set<std::string> expected_shell_names;
    for (size_t i = 1; i < num_blocks; ++i) {
      expected_shell_names.insert("Shell" + std::to_string(i - 1));
    }
    CHECK(block_groups.at("Shells") == expected_shell_names);
  }

  CHECK(angular_disk.initial_extents() == expected_extents);
  CHECK(angular_disk.initial_refinement_levels() == expected_refinement_level);

  // Create expected coordinate maps
  using Affine = CoordinateMaps::Affine;
  using Identity2D = CoordinateMaps::Identity<1>;
  using ProductMap = CoordinateMaps::ProductOf2Maps<Affine, Identity2D>;
  using PolarToCartesian = CoordinateMaps::PolarToCartesian;

  auto coord_maps = make_vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 2>>>();

  // Center block
  double inner_radius = 0.0;
  double current_outer_radius =
      radial_partitioning.empty() ? outer_radius : radial_partitioning[0];
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          ProductMap{Affine{-1.0, 1.0, inner_radius, current_outer_radius},
                     Identity2D{}},
          PolarToCartesian{}));

  // Shell blocks
  for (size_t i = 1; i < num_blocks; ++i) {
    inner_radius = radial_partitioning[i - 1];
    current_outer_radius =
        (i == num_blocks - 1) ? outer_radius : radial_partitioning[i];
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            ProductMap{Affine{-1.0, 1.0, inner_radius, current_outer_radius},
                       Identity2D{}},
            PolarToCartesian{}));
  }

  test_domain_construction(domain, expected_block_neighbors,
                           expected_external_boundaries, coord_maps, 10.0,
                           angular_disk.functions_of_time(),
                           expected_grid_to_inertial_maps);
  TestHelpers::domain::creators::test_functions_of_time(
      angular_disk, expected_functions_of_time, initial_expiration_times);
}

void test_angular_disk_single_block() {
  INFO("AngularDisk single block (no radial partitioning)");

  const double outer_radius = 2.0;
  const std::vector<double> radial_partitioning{};
  const size_t initial_disk_theta_grid_points = 5;  // Must be odd
  const std::vector<std::array<size_t, 2>> initial_annulus_grid_points{};
  const std::vector<std::array<size_t, 2>> refinement_level{{0, 0}};

  // Calculate expected extents for center disk
  const size_t theta_M = initial_disk_theta_grid_points / 2;
  const size_t disk_n_r = theta_M / 2 + 1 + theta_M % 2;
  const std::vector<std::array<size_t, 2>> expected_extents{
      {{disk_n_r, initial_disk_theta_grid_points}}};

  // Single block has no neighbors
  const std::vector<DirectionMap<2, BlockNeighbors<2>>> expected_neighbors{{}};

  // Single block has all external boundaries
  const std::vector<std::unordered_set<Direction<2>>> expected_externals{
      {Direction<2>::upper_xi()}};

  {
    INFO("Without boundary conditions");
    const creators::AngularDisk angular_disk{
        outer_radius, radial_partitioning, initial_disk_theta_grid_points,
        initial_annulus_grid_points};  // Uses all defaults

    test_angular_disk_construction(
        angular_disk, outer_radius, radial_partitioning, expected_extents,
        refinement_level, expected_neighbors, expected_externals);
  }

  {
    INFO("With boundary conditions");
    const creators::AngularDisk angular_disk{outer_radius,
                                             radial_partitioning,
                                             initial_disk_theta_grid_points,
                                             initial_annulus_grid_points,
                                             nullptr,
                                             create_boundary_condition()};

    test_angular_disk_construction(
        angular_disk, outer_radius, radial_partitioning, expected_extents,
        refinement_level, expected_neighbors, expected_externals, {}, {}, true);
  }
}

void test_angular_disk_multi_block() {
  INFO("AngularDisk multi-block (with radial partitioning)");

  const double outer_radius = 3.0;
  const std::vector<double> radial_partitioning{1.0, 2.0};
  const size_t initial_disk_theta_grid_points = 7;  // Must be odd
  const std::vector<std::array<size_t, 2>> initial_annulus_grid_points{
      {{4, 7}}, {{5, 9}}};  // Grid points for each shell
  const std::vector<std::array<size_t, 2>> refinement_level{{0, 0}};

  // Calculate expected extents
  const size_t theta_M = initial_disk_theta_grid_points / 2;
  const size_t disk_n_r = theta_M / 2 + 1 + theta_M % 2;
  const std::vector<std::array<size_t, 2>> expected_extents{
      {{disk_n_r, initial_disk_theta_grid_points}},  // Center
      {{4, 7}},                                      // Shell0
      {{5, 9}}                                       // Shell1
  };

  const OrientationMap<2> aligned = OrientationMap<2>::create_aligned();

  const std::vector<DirectionMap<2, BlockNeighbors<2>>> expected_neighbors{
      {{Direction<2>::upper_xi(), {1, aligned}}},
      {{Direction<2>::lower_xi(), {0, aligned}},
       {Direction<2>::upper_xi(), {2, aligned}}},
      {{Direction<2>::lower_xi(), {1, aligned}}}};

  const std::vector<std::unordered_set<Direction<2>>> expected_externals{
      {}, {}, {Direction<2>::upper_xi()}};

  {
    INFO("Multi-block without boundary conditions");
    const creators::AngularDisk angular_disk{
        outer_radius,
        radial_partitioning,
        initial_disk_theta_grid_points,
        initial_annulus_grid_points};

    test_angular_disk_construction(
        angular_disk, outer_radius, radial_partitioning, expected_extents,
        {refinement_level[0], refinement_level[0], refinement_level[0]},
        expected_neighbors, expected_externals);
  }

  {
    INFO("Multi-block with boundary conditions");
    const creators::AngularDisk angular_disk{outer_radius,
                                             radial_partitioning,
                                             initial_disk_theta_grid_points,
                                             initial_annulus_grid_points,
                                             nullptr,
                                             create_boundary_condition()};

    test_angular_disk_construction(
        angular_disk, outer_radius, radial_partitioning, expected_extents,
        {refinement_level[0], refinement_level[0], refinement_level[0]},
        expected_neighbors, expected_externals, {}, {}, true);
  }
}

void test_angular_disk_single_annulus_spec() {
  INFO("AngularDisk with single annulus grid specification applied to all");

  const double outer_radius = 2.5;
  const std::vector<double> radial_partitioning{1.5};
  const size_t initial_disk_theta_grid_points = 9;
  const std::array<size_t, 2> single_annulus_spec{
      {6, 11}};  // Will be applied to all shells
  const std::vector<std::array<size_t, 2>> refinement_level{{0, 0}};

  const size_t theta_M = initial_disk_theta_grid_points / 2;
  const size_t disk_n_r = theta_M / 2 + 1 + theta_M % 2;
  const std::vector<std::array<size_t, 2>> expected_extents{
      {{disk_n_r, initial_disk_theta_grid_points}}, {{6, 11}}};

  const OrientationMap<2> aligned = OrientationMap<2>::create_aligned();
  const std::vector<DirectionMap<2, BlockNeighbors<2>>> expected_neighbors{
      {{Direction<2>::upper_xi(), {1, aligned}}},
      {{Direction<2>::lower_xi(), {0, aligned}}}};

  const std::vector<std::unordered_set<Direction<2>>> expected_externals{
      {}, {Direction<2>::upper_xi()}};

  const creators::AngularDisk angular_disk{outer_radius, radial_partitioning,
                                           initial_disk_theta_grid_points,
                                           single_annulus_spec};

  test_angular_disk_construction(angular_disk, outer_radius,
                                 radial_partitioning, expected_extents,
                                 {refinement_level[0], refinement_level[0]},
                                 expected_neighbors, expected_externals);
}

void test_angular_disk_errors() {
  INFO("AngularDisk error conditions");

  const double outer_radius = 2.0;
  const size_t initial_disk_theta_grid_points = 5;
  const std::vector<std::array<size_t, 2>> initial_annulus_grid_points_empty{};
  const std::vector<std::array<size_t, 2>> initial_annulus_grid_points{{4, 7}};

  // Test unsorted radial partitioning
  CHECK_THROWS_WITH(
      creators::AngularDisk(outer_radius, {1.5, 1.0},
                            initial_disk_theta_grid_points,
                            initial_annulus_grid_points, nullptr, nullptr,
                            Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Specify radial partitioning in ascending order"));

  // Test radial partition >= outer_radius
  CHECK_THROWS_WITH(
      creators::AngularDisk(outer_radius, {2.0}, initial_disk_theta_grid_points,
                            initial_annulus_grid_points, nullptr, nullptr,
                            Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Last radial partition must be smaller than the outer radius"));

  // Test duplicate radial partitioning values
  CHECK_THROWS_WITH(
      creators::AngularDisk(outer_radius, {1.0, 1.0},
                            initial_disk_theta_grid_points,
                            initial_annulus_grid_points, nullptr, nullptr,
                            Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Radial partitioning contains duplicate element"));

  // Test outer boundary <= 0
  CHECK_THROWS_WITH(
      creators::AngularDisk(-1.0, {1.5}, initial_disk_theta_grid_points,
                            initial_annulus_grid_points, nullptr, nullptr,
                            Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("Must have a positive outer radius"));

  // Test radial partitioning <= 0
  CHECK_THROWS_WITH(
      creators::AngularDisk(outer_radius, {0.0}, initial_disk_theta_grid_points,
                            initial_annulus_grid_points, nullptr, nullptr,
                            Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("Radial partitions must be positive"));

  // Test odd disk theta
  CHECK_THROWS_WITH(creators::AngularDisk(
                        outer_radius, {1.5}, 6, initial_annulus_grid_points,
                        nullptr, nullptr, Options::Context{false, {}, 1, 1}),
                    Catch::Matchers::ContainsSubstring(
                        "The number of angular grid points must be odd"));

  // Test odd shell theta
  CHECK_THROWS_WITH(
      creators::AngularDisk(outer_radius, {1.5}, initial_disk_theta_grid_points,
                            std::vector<std::array<size_t, 2>>{{4, 4}}, nullptr,
                            nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "The number of angular grid points must be odd"));

  // Test None boundary condition
  auto none_bc = std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestNoneBoundaryCondition<2>>();
  CHECK_THROWS_WITH(creators::AngularDisk(
                        outer_radius, {}, initial_disk_theta_grid_points,
                        initial_annulus_grid_points_empty, nullptr,
                        std::move(none_bc), Options::Context{false, {}, 1, 1}),
                    Catch::Matchers::ContainsSubstring(
                        "None boundary condition is not supported"));

  // Test periodic boundary condition (not allowed for disk)
  auto periodic_bc = std::make_unique<TestHelpers::domain::BoundaryConditions::
                                          TestPeriodicBoundaryCondition<2>>();
  CHECK_THROWS_WITH(
      creators::AngularDisk(outer_radius, {}, initial_disk_theta_grid_points,
                            initial_annulus_grid_points_empty, nullptr,
                            std::move(periodic_bc),
                            Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("Cannot have periodic boundary"));

  // Test annulus grid points size mismatch
  CHECK_THROWS_WITH(
      creators::AngularDisk(
          outer_radius, {1.0, 1.5}, initial_disk_theta_grid_points,
          std::vector<std::array<size_t, 2>>{
              {4, 7}},  // Only one spec for two shells
          nullptr, nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("must be one larger than"));
}

void test_angular_disk_factory() {
  INFO("AngularDisk factory tests");

  {
    INFO("Factory without boundary conditions");
    const auto angular_disk =
        TestHelpers::test_option_tag<domain::OptionTags::DomainCreator<2>,
                                     TestHelpers::domain::BoundaryConditions::
                                         MetavariablesWithoutBoundaryConditions<
                                             2, domain::creators::AngularDisk>>(
            "AngularDisk:\n"
            "  OuterRadius: 2.5\n"
            "  RadialPartitioning: [1.0]\n"
            "  InitialDiskThetaGridPoints: 7\n"
            "  InitialAnnulusGridPoints: [4, 9]\n"
            "  TimeDependence: None\n");

    const auto* disk_creator =
        dynamic_cast<const creators::AngularDisk*>(angular_disk.get());

    // Test basic properties
    CHECK(disk_creator->block_names() ==
          std::vector<std::string>{"InnerDisk", "Shell0"});

    const size_t theta_M = 7 / 2;
    const size_t disk_n_r = theta_M / 2 + 1 + theta_M % 2;
    CHECK(disk_creator->initial_extents() ==
          std::vector<std::array<size_t, 2>>{{{disk_n_r, 7}}, {{4, 9}}});
  }

  {
    INFO("Factory with boundary conditions");
    const auto angular_disk =
        TestHelpers::test_option_tag<domain::OptionTags::DomainCreator<2>,
                                     TestHelpers::domain::BoundaryConditions::
                                         MetavariablesWithBoundaryConditions<
                                             2, domain::creators::AngularDisk>>(
            "AngularDisk:\n"
            "  OuterRadius: 3.0\n"
            "  RadialPartitioning: []\n"
            "  InitialDiskThetaGridPoints: 5\n"
            "  InitialAnnulusGridPoints: []\n"
            "  TimeDependence: None\n"
            "  BoundaryCondition:\n"
            "    TestBoundaryCondition:\n"
            "      Direction: lower-xi\n"
            "      BlockId: 0\n");

    const auto* disk_creator =
        dynamic_cast<const creators::AngularDisk*>(angular_disk.get());
    CHECK(disk_creator->block_names() ==
          std::vector<std::string>{"InnerDisk"});
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.AngularDisk", "[Domain][Unit]") {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  test_angular_disk_single_block();
  test_angular_disk_multi_block();
  test_angular_disk_single_annulus_spec();
  test_angular_disk_errors();
  test_angular_disk_factory();
}

}  // namespace domain
