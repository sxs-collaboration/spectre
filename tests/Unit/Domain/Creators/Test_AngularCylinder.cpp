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
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/PolarToCartesian.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/AngularCylinder.hpp"
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
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::lower_xi(), 0);
}

template <typename... FuncsOfTime>
void test_angular_cylinder_construction(
    const creators::AngularCylinder& angular_cylinder,
    const double outer_radius, const double lower_z_bound,
    const double upper_z_bound, const std::vector<double>& radial_partitioning,
    const std::vector<double>& partitioning_in_z,
    const std::vector<domain::CoordinateMaps::Distribution>& distribution_in_z,
    const std::vector<std::array<size_t, 3>>& expected_extents,
    const std::vector<std::array<size_t, 3>>& expected_refinement_level,
    const std::vector<DirectionMap<3, BlockNeighbors<3>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<Direction<3>>>&
        expected_external_boundaries,
    const bool is_periodic_in_z = false,
    const std::tuple<std::pair<std::string, FuncsOfTime>...>&
        expected_functions_of_time = {},
    const std::vector<std::unique_ptr<domain::CoordinateMapBase<
        Frame::Grid, Frame::Inertial, 3>>>& expected_grid_to_inertial_maps = {},
    const bool expect_boundary_conditions = false,
    const std::unordered_map<std::string, double>& initial_expiration_times =
        {}) {
  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      angular_cylinder, expect_boundary_conditions, is_periodic_in_z);
  CHECK(angular_cylinder.grid_anchors().empty());

  // Calculate expected dimensions
  const size_t num_radial_blocks = 1 + radial_partitioning.size();
  const size_t num_layers = 1 + partitioning_in_z.size();

  // Verify block names
  std::vector<std::string> expected_block_names;
  for (size_t layer = 0; layer < num_layers; ++layer) {
    const std::string layer_prefix =
        num_layers > 1 ? "Layer" + std::to_string(layer) : "";

    // Center disk
    expected_block_names.emplace_back(layer_prefix + "CenterDisk");

    // Shells
    for (size_t shell = 1; shell < num_radial_blocks; ++shell) {
      expected_block_names.emplace_back(layer_prefix + "Shell" +
                                        std::to_string(shell - 1));
    }
  }
  CHECK(angular_cylinder.block_names() == expected_block_names);

  // Verify block groups
  const auto block_groups = angular_cylinder.block_groups();

  // Per-layer groups (only if num_layers > 1)
  if (num_layers > 1) {
    for (size_t layer = 0; layer < num_layers; ++layer) {
      const std::string layer_name = "Layer" + std::to_string(layer);
      CHECK(block_groups.contains(layer_name));

      std::unordered_set<std::string> expected_layer_blocks;
      for (size_t radial = 0; radial < num_radial_blocks; ++radial) {
        expected_layer_blocks.insert(
            expected_block_names[layer * num_radial_blocks + radial]);
      }
      CHECK(block_groups.at(layer_name) == expected_layer_blocks);
    }
  }

  // Global groups
  CHECK(block_groups.contains("InnerDisks"));
  std::unordered_set<std::string> expected_center_blocks;
  for (size_t layer = 0; layer < num_layers; ++layer) {
    expected_center_blocks.insert(
        expected_block_names[layer * num_radial_blocks]);
  }
  CHECK(block_groups.at("InnerDisks") == expected_center_blocks);

  if (num_radial_blocks > 1) {
    CHECK(block_groups.contains("Shells"));
    std::unordered_set<std::string> expected_shell_blocks;
    for (size_t layer = 0; layer < num_layers; ++layer) {
      for (size_t shell = 1; shell < num_radial_blocks; ++shell) {
        expected_shell_blocks.insert(
            expected_block_names[layer * num_radial_blocks + shell]);
      }
    }
    CHECK(block_groups.at("Shells") == expected_shell_blocks);
  } else {
    CHECK(not block_groups.contains("Shells"));
  }

  CHECK(angular_cylinder.initial_extents() == expected_extents);
  CHECK(angular_cylinder.initial_refinement_levels() ==
        expected_refinement_level);

  // Create expected coordinate maps
  using Affine = CoordinateMaps::Affine;
  using Identity1D = CoordinateMaps::Identity<1>;
  using Interval = CoordinateMaps::Interval;
  using ProductMap3D =
      CoordinateMaps::ProductOf3Maps<Affine, Identity1D, Interval>;
  using PolarToCartesian = CoordinateMaps::PolarToCartesian;
  using PolarProduct =
      CoordinateMaps::ProductOf2Maps<PolarToCartesian, Identity1D>;

  auto coord_maps = make_vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>();

  // Create coordinate maps for each block
  for (size_t layer = 0; layer < num_layers; ++layer) {
    const double z_lower =
        (layer == 0) ? lower_z_bound : partitioning_in_z[layer - 1];
    const double z_upper =
        (layer == num_layers - 1) ? upper_z_bound : partitioning_in_z[layer];

    for (size_t radial = 0; radial < num_radial_blocks; ++radial) {
      const double inner_radius =
          (radial == 0) ? 0.0 : radial_partitioning[radial - 1];
      const double current_outer_radius = (radial == num_radial_blocks - 1)
                                              ? outer_radius
                                              : radial_partitioning[radial];

      coord_maps.emplace_back(
          make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
              ProductMap3D{
                  Affine{-1.0, 1.0, inner_radius, current_outer_radius},
                  Identity1D{},
                  Interval{-1.0, 1.0, z_lower, z_upper,
                           distribution_in_z[layer]}},
              PolarProduct{PolarToCartesian{}, Identity1D{}}));
    }
  }

  test_domain_construction(domain, expected_block_neighbors,
                           expected_external_boundaries, coord_maps, 10.0,
                           angular_cylinder.functions_of_time(),
                           expected_grid_to_inertial_maps);
  TestHelpers::domain::creators::test_functions_of_time(
      angular_cylinder, expected_functions_of_time, initial_expiration_times);
}

void test_angular_cylinder_single_block() {
  INFO("AngularCylinder single block (no partitioning)");

  const double outer_radius = 2.0;
  const double lower_z_bound = -1.0;
  const double upper_z_bound = 1.0;
  const std::vector<double> radial_partitioning{};
  const std::vector<double> partitioning_in_z{};
  const size_t initial_cylinder_theta_grid_points = 5;
  const size_t initial_cylinder_z_grid_points = 4;
  const std::vector<std::array<size_t, 3>>
      initial_hollow_cylinder_grid_points{};
  const std::vector<domain::CoordinateMaps::Distribution> distribution_in_z{
      domain::CoordinateMaps::Distribution::Linear};

  // Calculate expected extents for center disk
  const size_t theta_M = initial_cylinder_theta_grid_points / 2;
  const size_t disk_n_r = theta_M / 2 + 1 + theta_M % 2;
  const std::vector<std::array<size_t, 3>> expected_extents{
      {{disk_n_r, initial_cylinder_theta_grid_points,
        initial_cylinder_z_grid_points}}};

  const std::vector<std::array<size_t, 3>> refinement_level{{{0, 0, 2}}};

  // Single block has no neighbors
  const std::vector<DirectionMap<3, BlockNeighbors<3>>> expected_neighbors{{}};

  // Single block has external boundaries in z and radial directions
  const std::vector<std::unordered_set<Direction<3>>> expected_externals{
      {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
       Direction<3>::upper_zeta()}};

  {
    INFO("With boundary conditions");
    const creators::AngularCylinder angular_cylinder{
        outer_radius,
        lower_z_bound,
        upper_z_bound,
        radial_partitioning,
        partitioning_in_z,
        initial_cylinder_theta_grid_points,
        initial_cylinder_z_grid_points,
        initial_hollow_cylinder_grid_points,
        distribution_in_z,
        2_st,
        nullptr,
        create_boundary_condition(),
        create_boundary_condition(),
        create_boundary_condition()};

    test_angular_cylinder_construction(
        angular_cylinder, outer_radius, lower_z_bound, upper_z_bound,
        radial_partitioning, partitioning_in_z, distribution_in_z,
        expected_extents, refinement_level, expected_neighbors,
        expected_externals, false, {}, {}, true);
  }
}

void test_angular_cylinder_multi_radial() {
  INFO("AngularCylinder multiple radial blocks");

  const double outer_radius = 3.0;
  const double lower_z_bound = 0.0;
  const double upper_z_bound = 2.0;
  const std::vector<double> radial_partitioning{1.0, 2.0};
  const std::vector<double> partitioning_in_z{};
  const size_t initial_cylinder_theta_grid_points = 7;
  const size_t initial_cylinder_z_grid_points = 6;
  const std::vector<std::array<size_t, 3>> initial_hollow_cylinder_grid_points{
      {{4, 7, 5}}, {{5, 9, 6}}};
  const std::vector<domain::CoordinateMaps::Distribution> distribution_in_z{
      domain::CoordinateMaps::Distribution::Linear};

  // Calculate expected extents
  const size_t theta_M = initial_cylinder_theta_grid_points / 2;
  const size_t disk_n_r = theta_M / 2 + 1 + theta_M % 2;
  const std::vector<std::array<size_t, 3>> expected_extents{
      {{disk_n_r, initial_cylinder_theta_grid_points,
        initial_cylinder_z_grid_points}},
      {{4, 7, 5}},
      {{5, 9, 6}}};

  const std::vector<std::array<size_t, 3>> refinement_level{{{0, 0, 1}}};

  const OrientationMap<3> aligned = OrientationMap<3>::create_aligned();

  const std::vector<DirectionMap<3, BlockNeighbors<3>>> expected_neighbors{
      {{Direction<3>::upper_xi(), {1, aligned}}},
      {{Direction<3>::lower_xi(), {0, aligned}},
       {Direction<3>::upper_xi(), {2, aligned}}},
      {{Direction<3>::lower_xi(), {1, aligned}}}};

  const std::vector<std::unordered_set<Direction<3>>> expected_externals{
      {Direction<3>::lower_zeta(), Direction<3>::upper_zeta()},
      {Direction<3>::lower_zeta(), Direction<3>::upper_zeta()},
      {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
       Direction<3>::upper_zeta()}};

  const creators::AngularCylinder angular_cylinder{
      outer_radius,
      lower_z_bound,
      upper_z_bound,
      radial_partitioning,
      partitioning_in_z,
      initial_cylinder_theta_grid_points,
      initial_cylinder_z_grid_points,
      initial_hollow_cylinder_grid_points,
      distribution_in_z,
      1_st,
      nullptr,
      create_boundary_condition(),
      create_boundary_condition(),
      create_boundary_condition()};

  test_angular_cylinder_construction(
      angular_cylinder, outer_radius, lower_z_bound, upper_z_bound,
      radial_partitioning, partitioning_in_z, distribution_in_z,
      expected_extents,
      {refinement_level[0], refinement_level[0], refinement_level[0]},
      expected_neighbors, expected_externals, false, {}, {}, true);
}

void test_angular_cylinder_multi_layer() {
  INFO("AngularCylinder multiple layers");

  const double outer_radius = 2.0;
  const double lower_z_bound = -2.0;
  const double upper_z_bound = 2.0;
  const std::vector<double> radial_partitioning{1.0};
  const std::vector<double> partitioning_in_z{0.0};
  const size_t initial_cylinder_theta_grid_points = 5;
  const size_t initial_cylinder_z_grid_points = 4;
  const std::array<size_t, 3> single_hollow_spec{{3, 5, 5}};
  const std::vector<domain::CoordinateMaps::Distribution> distribution_in_z{
      domain::CoordinateMaps::Distribution::Linear,
      domain::CoordinateMaps::Distribution::Linear};

  const size_t theta_M = initial_cylinder_theta_grid_points / 2;
  const size_t disk_n_r = theta_M / 2 + 1 + theta_M % 2;
  const std::vector<std::array<size_t, 3>> expected_extents{
      {{disk_n_r, initial_cylinder_theta_grid_points,
        initial_cylinder_z_grid_points}},
      {{3, 5, 5}},
      {{disk_n_r, initial_cylinder_theta_grid_points,
        initial_cylinder_z_grid_points}},
      {{3, 5, 5}}};

  const std::vector<std::array<size_t, 3>> refinement_level{{{0, 0, 0}}};

  const OrientationMap<3> aligned = OrientationMap<3>::create_aligned();

  const std::vector<DirectionMap<3, BlockNeighbors<3>>> expected_neighbors{
      {{Direction<3>::upper_xi(), {1, aligned}},
       {Direction<3>::upper_zeta(), {2, aligned}}},
      {{Direction<3>::lower_xi(), {0, aligned}},
       {Direction<3>::upper_zeta(), {3, aligned}}},
      {{Direction<3>::upper_xi(), {3, aligned}},
       {Direction<3>::lower_zeta(), {0, aligned}}},
      {{Direction<3>::lower_xi(), {2, aligned}},
       {Direction<3>::lower_zeta(), {1, aligned}}}};

  const std::vector<std::unordered_set<Direction<3>>> expected_externals{
      {Direction<3>::lower_zeta()},
      {Direction<3>::upper_xi(), Direction<3>::lower_zeta()},
      {Direction<3>::upper_zeta()},
      {Direction<3>::upper_xi(), Direction<3>::upper_zeta()}};

  const creators::AngularCylinder angular_cylinder{
      outer_radius,
      lower_z_bound,
      upper_z_bound,
      radial_partitioning,
      partitioning_in_z,
      initial_cylinder_theta_grid_points,
      initial_cylinder_z_grid_points,
      single_hollow_spec,
      distribution_in_z,
      std::vector<size_t>{0, 1},
      nullptr,
      create_boundary_condition(),
      create_boundary_condition(),
      create_boundary_condition()};

  test_angular_cylinder_construction(
      angular_cylinder, outer_radius, lower_z_bound, upper_z_bound,
      radial_partitioning, partitioning_in_z, distribution_in_z,
      expected_extents, {{{0, 0, 0}}, {{0, 0, 0}}, {{0, 0, 1}}, {{0, 0, 1}}},
      expected_neighbors, expected_externals, false, {}, {}, true);
}

void test_angular_cylinder_periodic_z() {
  INFO("AngularCylinder with periodic Z");

  const double outer_radius = 1.5;
  const double lower_z_bound = 0.0;
  const double upper_z_bound = 1.0;
  const std::vector<double> radial_partitioning{};
  const std::vector<double> partitioning_in_z{0.5};
  const size_t initial_cylinder_theta_grid_points = 3;
  const size_t initial_cylinder_z_grid_points = 3;
  const std::vector<std::array<size_t, 3>>
      initial_hollow_cylinder_grid_points{};
  const std::vector<domain::CoordinateMaps::Distribution> distribution_in_z{
      domain::CoordinateMaps::Distribution::Linear,
      domain::CoordinateMaps::Distribution::Linear};

  const size_t theta_M = initial_cylinder_theta_grid_points / 2;
  const size_t disk_n_r = theta_M / 2 + 1 + theta_M % 2;
  const std::vector<std::array<size_t, 3>> expected_extents{
      {{disk_n_r, initial_cylinder_theta_grid_points,
        initial_cylinder_z_grid_points}},
      {{disk_n_r, initial_cylinder_theta_grid_points,
        initial_cylinder_z_grid_points}}};

  const std::vector<std::array<size_t, 3>> refinement_level{{{0, 0, 3}}};

  const OrientationMap<3> aligned = OrientationMap<3>::create_aligned();

  const std::vector<DirectionMap<3, BlockNeighbors<3>>> expected_neighbors{
      {{Direction<3>::upper_zeta(), {1, aligned}},
       {Direction<3>::lower_zeta(), {1, aligned}}},
      {{Direction<3>::lower_zeta(), {0, aligned}},
       {Direction<3>::upper_zeta(), {0, aligned}}}};

  const std::vector<std::unordered_set<Direction<3>>> expected_externals{
      {Direction<3>::upper_xi()}, {Direction<3>::upper_xi()}};

  auto periodic_bc = std::make_unique<TestHelpers::domain::BoundaryConditions::
                                          TestPeriodicBoundaryCondition<3>>();
  auto periodic_bc2 = std::make_unique<TestHelpers::domain::BoundaryConditions::
                                           TestPeriodicBoundaryCondition<3>>();

  const creators::AngularCylinder angular_cylinder{
      outer_radius,
      lower_z_bound,
      upper_z_bound,
      radial_partitioning,
      partitioning_in_z,
      initial_cylinder_theta_grid_points,
      initial_cylinder_z_grid_points,
      initial_hollow_cylinder_grid_points,
      distribution_in_z,
      std::vector<size_t>{3, 2},
      nullptr,
      std::move(periodic_bc),
      std::move(periodic_bc2),
      nullptr};

  test_angular_cylinder_construction(
      angular_cylinder, outer_radius, lower_z_bound, upper_z_bound,
      radial_partitioning, partitioning_in_z, distribution_in_z,
      expected_extents, {{{0, 0, 3}}, {{0, 0, 2}}}, expected_neighbors,
      expected_externals, true);
}

void test_angular_cylinder_errors() {
  INFO("AngularCylinder error conditions");

  const double outer_radius = 2.0;
  const double lower_z_bound = 0.0;
  const double upper_z_bound = 1.0;
  const size_t initial_cylinder_theta_grid_points = 5;
  const size_t initial_cylinder_z_grid_points = 4;
  const std::vector<std::array<size_t, 3>>
      initial_hollow_cylinder_grid_points_empty{};
  const std::vector<std::array<size_t, 3>> initial_hollow_cylinder_grid_points{
      {4, 7, 5}};

  // Test outer_radius <= 0
  CHECK_THROWS_WITH(
      creators::AngularCylinder(
          0.0, lower_z_bound, upper_z_bound, {}, {},
          initial_cylinder_theta_grid_points, initial_cylinder_z_grid_points,
          initial_hollow_cylinder_grid_points_empty,
          {domain::CoordinateMaps::Distribution::Linear}, 0_st, nullptr, false,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("OuterRadius must be positive"));

  // Test lower_z_bound >= upper_z_bound
  CHECK_THROWS_WITH(
      creators::AngularCylinder(
          outer_radius, 1.0, 0.0, {}, {}, initial_cylinder_theta_grid_points,
          initial_cylinder_z_grid_points,
          initial_hollow_cylinder_grid_points_empty,
          {domain::CoordinateMaps::Distribution::Linear}, 0_st, nullptr, false,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("LowerZBound must be less than"));

  // Test distribution_in_z size mismatch
  CHECK_THROWS_WITH(
      creators::AngularCylinder(
          outer_radius, lower_z_bound, upper_z_bound, {}, {0.5},  // Two layers
          initial_cylinder_theta_grid_points, initial_cylinder_z_grid_points,
          initial_hollow_cylinder_grid_points_empty,
          {domain::CoordinateMaps::Distribution::Linear}, 0_st, nullptr, false,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Specify a 'DistributionInZ' for every layer"));

  // Test first distribution not Linear
  CHECK_THROWS_WITH(
      creators::AngularCylinder(
          outer_radius, lower_z_bound, upper_z_bound, {}, {},
          initial_cylinder_theta_grid_points, initial_cylinder_z_grid_points,
          initial_hollow_cylinder_grid_points_empty,
          {domain::CoordinateMaps::Distribution::Logarithmic}, 0_st, nullptr,
          false, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "must be 'Linear' for the lowermost layer"));

  // Test radial partitioning <= 0
  CHECK_THROWS_WITH(
      creators::AngularCylinder(
          outer_radius, lower_z_bound, upper_z_bound, {0.0}, {},
          initial_cylinder_theta_grid_points, initial_cylinder_z_grid_points,
          initial_hollow_cylinder_grid_points,
          {domain::CoordinateMaps::Distribution::Linear}, 0_st, nullptr, false,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("Radial partitions must be positive"));

  // Test odd center cylinder theta
  CHECK_THROWS_WITH(creators::AngularCylinder(
                        outer_radius, lower_z_bound, upper_z_bound, {}, {}, 6,
                        initial_cylinder_z_grid_points,  // Even number
                        initial_hollow_cylinder_grid_points_empty,
                        {domain::CoordinateMaps::Distribution::Linear}, 0_st,
                        nullptr, false, Options::Context{false, {}, 1, 1}),
                    Catch::Matchers::ContainsSubstring(
                        "The number of angular grid points must be odd"));

  // Test odd shell theta (single specification)
  CHECK_THROWS_WITH(
      creators::AngularCylinder(
          outer_radius, lower_z_bound, upper_z_bound, {1.5}, {},
          initial_cylinder_theta_grid_points, initial_cylinder_z_grid_points,
          std::array<size_t, 3>{{4, 6, 5}},  // Even theta
          {domain::CoordinateMaps::Distribution::Linear}, 0_st, nullptr, false,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "The number of angular grid points must be odd"));

  // Test odd shell theta (vector specification)
  CHECK_THROWS_WITH(
      creators::AngularCylinder(
          outer_radius, lower_z_bound, upper_z_bound, {1.5}, {},
          initial_cylinder_theta_grid_points, initial_cylinder_z_grid_points,
          std::vector<std::array<size_t, 3>>{{4, 6, 5}},  // Even theta
          {domain::CoordinateMaps::Distribution::Linear}, 0_st, nullptr, false,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "The number of angular grid points must be odd"));

  // Test None boundary conditions
  auto none_bc = std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestNoneBoundaryCondition<3>>();
  CHECK_THROWS_WITH(
      creators::AngularCylinder(
          outer_radius, lower_z_bound, upper_z_bound, {}, {},
          initial_cylinder_theta_grid_points, initial_cylinder_z_grid_points,
          initial_hollow_cylinder_grid_points_empty,
          {domain::CoordinateMaps::Distribution::Linear}, 0_st, nullptr,
          std::move(none_bc), nullptr, nullptr),
      Catch::Matchers::ContainsSubstring(
          "None boundary condition is not supported"));

  // Test mismatched periodic boundary conditions
  auto periodic_bc = std::make_unique<TestHelpers::domain::BoundaryConditions::
                                          TestPeriodicBoundaryCondition<3>>();
  auto regular_bc = create_boundary_condition();
  CHECK_THROWS_WITH(
      creators::AngularCylinder(
          outer_radius, lower_z_bound, upper_z_bound, {}, {},
          initial_cylinder_theta_grid_points, initial_cylinder_z_grid_points,
          initial_hollow_cylinder_grid_points_empty,
          {domain::CoordinateMaps::Distribution::Linear}, 0_st, nullptr,
          std::move(periodic_bc), std::move(regular_bc), nullptr),
      Catch::Matchers::ContainsSubstring("Either both lower and upper"));

  // Test periodic mantle boundary condition
  auto periodic_mantle_bc =
      std::make_unique<TestHelpers::domain::BoundaryConditions::
                           TestPeriodicBoundaryCondition<3>>();
  CHECK_THROWS_WITH(
      creators::AngularCylinder(
          outer_radius, lower_z_bound, upper_z_bound, {}, {},
          initial_cylinder_theta_grid_points, initial_cylinder_z_grid_points,
          initial_hollow_cylinder_grid_points_empty,
          {domain::CoordinateMaps::Distribution::Linear}, 0_st, nullptr,
          nullptr, nullptr, std::move(periodic_mantle_bc)),
      Catch::Matchers::ContainsSubstring(
          "A cylinder can't have periodic boundary"));

  // Test nullptr mantle boundary condition but setting z
  auto regular_z_lower_bc = create_boundary_condition();
  auto regular_z_upper_bc = create_boundary_condition();
  CHECK_THROWS_WITH(
      creators::AngularCylinder(outer_radius, lower_z_bound, upper_z_bound, {},
                                {}, initial_cylinder_theta_grid_points,
                                initial_cylinder_z_grid_points,
                                initial_hollow_cylinder_grid_points_empty,
                                {domain::CoordinateMaps::Distribution::Linear},
                                0_st, nullptr, std::move(regular_z_lower_bc),
                                std::move(regular_z_upper_bc), nullptr),
      Catch::Matchers::ContainsSubstring(
          "Mantle boundary condition is not set, but lower is"));

  // Test InitialRefinementInZ size mismatch
  CHECK_THROWS_WITH(
      creators::AngularCylinder(
          outer_radius, lower_z_bound, upper_z_bound, {}, {0.5},  // Two layers
          initial_cylinder_theta_grid_points, initial_cylinder_z_grid_points,
          initial_hollow_cylinder_grid_points_empty,
          {domain::CoordinateMaps::Distribution::Linear,
           domain::CoordinateMaps::Distribution::Linear},
          std::vector<size_t>{1}, nullptr, false,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "InitialRefinementInZ must have one entry per layer"));
}

void test_angular_cylinder_factory() {
  INFO("AngularCylinder factory tests");

  {
    INFO("Factory with boundary conditions");
    const auto angular_cylinder = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<3>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithBoundaryConditions<
                3, domain::creators::AngularCylinder>>(
        "AngularCylinder:\n"
        "  OuterRadius: 1.5\n"
        "  LowerZBound: -1.0\n"
        "  UpperZBound: 1.0\n"
        "  RadialPartitioning: []\n"
        "  PartitioningInZ: []\n"
        "  InitialCylinderThetaGridPoints: 3\n"
        "  InitialCylinderZGridPoints: 3\n"
        "  InitialHollowCylinderGridPoints: []\n"
        "  DistributionInZ: [Linear]\n"
        "  InitialRefinementInZ: 1\n"
        "  TimeDependence: None\n"
        "  BoundaryConditions:\n"
        "    LowerZ:\n"
        "      TestBoundaryCondition:\n"
        "        Direction: lower-xi\n"
        "        BlockId: 0\n"
        "    UpperZ:\n"
        "      TestBoundaryCondition:\n"
        "        Direction: lower-xi\n"
        "        BlockId: 0\n"
        "    Mantle:\n"
        "      TestBoundaryCondition:\n"
        "        Direction: lower-xi\n"
        "        BlockId: 0\n");

    const auto* cylinder_creator =
        dynamic_cast<const creators::AngularCylinder*>(angular_cylinder.get());
    CHECK(cylinder_creator->block_names() ==
          std::vector<std::string>{"CenterDisk"});
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.AngularCylinder", "[Domain][Unit]") {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  test_angular_cylinder_single_block();
  test_angular_cylinder_multi_radial();
  test_angular_cylinder_multi_layer();
  test_angular_cylinder_periodic_z();
  test_angular_cylinder_errors();
  test_angular_cylinder_factory();
}

}  // namespace domain
