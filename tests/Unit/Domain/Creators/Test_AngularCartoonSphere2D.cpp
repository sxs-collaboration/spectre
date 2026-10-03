// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numbers>
#include <pup.h>
#include <string>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/PolarToCartesian.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/AngularCartoonSphere2D.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Options/Context.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeVector.hpp"

namespace {
using Distribution = domain::CoordinateMaps::Distribution;

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_boundary_condition() {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::lower_xi(), 0);
}

// Rebuilds the block coordinate maps independently of the creator so that
// `test_domain_construction` checks the maps rather than just their
// self-consistency.
std::vector<std::unique_ptr<
    domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>
build_coord_maps(const double inner_radius, const double outer_radius,
                 const std::vector<double>& radial_partitioning,
                 const std::vector<Distribution>& radial_distribution) {
  using Identity1D = domain::CoordinateMaps::Identity<1>;
  using Interval = domain::CoordinateMaps::Interval;
  const size_t num_radial_blocks = 1 + radial_partitioning.size();

  auto coord_maps = make_vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>();
  for (size_t radial = 0; radial < num_radial_blocks; ++radial) {
    const double inner_r =
        radial == 0 ? inner_radius : radial_partitioning[radial - 1];
    const double outer_r = radial == num_radial_blocks - 1
                               ? outer_radius
                               : radial_partitioning[radial];
    coord_maps.emplace_back(
        domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            domain::CoordinateMaps::ProductOf3Maps<Interval, Identity1D,
                                                   Identity1D>{
                Interval{-1.0, 1.0, inner_r, outer_r,
                         radial_distribution[radial], 0.0},
                Identity1D{}, Identity1D{}},
            domain::CoordinateMaps::ProductOf2Maps<
                domain::CoordinateMaps::PolarToCartesian, Identity1D>{
                domain::CoordinateMaps::PolarToCartesian{}, Identity1D{}},
            domain::CoordinateMaps::ProductOf2Maps<
                domain::CoordinateMaps::DiscreteRotation<2>, Identity1D>{
                domain::CoordinateMaps::DiscreteRotation<2>{OrientationMap<2>{
                    std::array<Direction<2>, 2>{{Direction<2>::upper_eta(),
                                                 Direction<2>::lower_xi()}}}},
                Identity1D{}}));
  }
  return coord_maps;
}

void check_topologies(const Domain<3>& domain) {
  for (const auto& block : domain.blocks()) {
    CHECK(block.topologies() == domain::topologies::cartoon_cylinder_annulus);
  }
}

void test_half_plane_geometry() {
  INFO("AngularCartoonSphere2D half-plane geometry");
  const double inner_radius = 1.0;
  const double outer_radius = 4.0;

  const domain::creators::AngularCartoonSphere2D creator{
      inner_radius,
      outer_radius,
      {},
      {Distribution::Linear},
      std::array<size_t, 2>{{4, 5}},
      std::vector<size_t>{0},
      nullptr,
      create_boundary_condition(),
      create_boundary_condition()};
  const auto domain = creator.create_domain();
  const auto& map = domain.blocks()[0].stationary_map();

  const double pi = std::numbers::pi;
  // (xi, phi) pairs with the expected (x, y). xi = -1 is the inner radius and
  // xi = +1 the outer radius; the third logical coordinate is ignored because
  // the Cartoon direction maps to z = 0 at the single grid point.
  const std::array<std::array<double, 4>, 6> checks{{
      // xi, phi, expected x, expected y
      {{-1.0, 0.0, 0.0, -inner_radius}},
      {{-1.0, 0.5 * pi, inner_radius, 0.0}},
      {{-1.0, pi, 0.0, inner_radius}},
      {{1.0, 0.0, 0.0, -outer_radius}},
      {{1.0, 0.5 * pi, outer_radius, 0.0}},
      {{1.0, 0.25 * pi, outer_radius / std::numbers::sqrt2,
        -outer_radius / std::numbers::sqrt2}},
  }};
  for (const auto& [xi, phi, expected_x, expected_y] : checks) {
    CAPTURE(xi);
    CAPTURE(phi);
    const auto inertial =
        map(tnsr::I<double, 3, Frame::BlockLogical>{{{xi, phi, 0.0}}});
    CHECK(get<0>(inertial) == approx(expected_x));
    CHECK(get<1>(inertial) == approx(expected_y));
    CHECK(get<2>(inertial) == approx(0.0));
  }

  // Every interior azimuthal collocation-like angle must stay in x >= 0, and
  // the radius must be preserved exactly by the angular map.
  for (size_t i = 1; i < 32; ++i) {
    const double phi = pi * static_cast<double>(i) / 32.0;
    CAPTURE(phi);
    const auto inertial =
        map(tnsr::I<double, 3, Frame::BlockLogical>{{{0.0, phi, 0.0}}});
    CHECK(get<0>(inertial) >= 0.0);
    const double radius =
        sqrt(square(get<0>(inertial)) + square(get<1>(inertial)));
    CHECK(radius == approx(0.5 * (inner_radius + outer_radius)));
  }
}

void test_single_block() {
  INFO("AngularCartoonSphere2D single block");
  const double inner_radius = 0.5;
  const double outer_radius = 2.0;
  const std::vector<double> radial_partitioning{};
  const std::vector<Distribution> radial_distribution{Distribution::Linear};

  const domain::creators::AngularCartoonSphere2D creator{
      inner_radius,
      outer_radius,
      radial_partitioning,
      radial_distribution,
      std::array<size_t, 2>{{4, 7}},
      std::vector<size_t>{2},
      nullptr,
      create_boundary_condition(),
      create_boundary_condition()};

  const auto domain =
      TestHelpers::domain::creators::test_domain_creator(creator, true);

  CHECK(creator.block_names() == std::vector<std::string>{"Shell0"});
  CHECK(creator.block_groups() ==
        std::unordered_map<std::string, std::unordered_set<std::string>>{
            {"Shells", {"Shell0"}}});
  CHECK(creator.initial_extents() ==
        std::vector<std::array<size_t, 3>>{{{4, 7, 1}}});
  CHECK(creator.initial_refinement_levels() ==
        std::vector<std::array<size_t, 3>>{{{2, 0, 0}}});
  CHECK(creator.functions_of_time().empty());
  CHECK(creator.grid_anchors().empty());
  check_topologies(domain);

  const std::vector<DirectionMap<3, BlockNeighbors<3>>> expected_neighbors{{}};
  const std::vector<std::unordered_set<Direction<3>>> expected_externals{
      {Direction<3>::lower_xi(), Direction<3>::upper_xi()}};

  test_domain_construction(
      domain, expected_neighbors, expected_externals,
      build_coord_maps(inner_radius, outer_radius, radial_partitioning,
                       radial_distribution));
}

void test_multiple_shells() {
  INFO("AngularCartoonSphere2D multiple shells");
  const double inner_radius = 1.0;
  const double outer_radius = 8.0;
  const std::vector<double> radial_partitioning{2.0, 4.0};
  const std::vector<Distribution> radial_distribution{
      Distribution::Linear, Distribution::Logarithmic, Distribution::Inverse};

  const domain::creators::AngularCartoonSphere2D creator{
      inner_radius,
      outer_radius,
      radial_partitioning,
      radial_distribution,
      std::vector<std::array<size_t, 2>>{{{3, 5}}, {{4, 6}}, {{5, 8}}},
      std::vector<size_t>{0, 1, 2},
      nullptr,
      create_boundary_condition(),
      create_boundary_condition()};

  const auto domain =
      TestHelpers::domain::creators::test_domain_creator(creator, true);

  CHECK(creator.block_names() ==
        std::vector<std::string>{"Shell0", "Shell1", "Shell2"});
  CHECK(creator.block_groups() ==
        std::unordered_map<std::string, std::unordered_set<std::string>>{
            {"Shells", {"Shell0", "Shell1", "Shell2"}}});
  // Per-shell grid points are not broadcast when a vector is given.
  CHECK(creator.initial_extents() ==
        std::vector<std::array<size_t, 3>>{
            {{3, 5, 1}}, {{4, 6, 1}}, {{5, 8, 1}}});
  CHECK(creator.initial_refinement_levels() ==
        std::vector<std::array<size_t, 3>>{
            {{0, 0, 0}}, {{1, 0, 0}}, {{2, 0, 0}}});
  check_topologies(domain);

  const auto aligned = OrientationMap<3>::create_aligned();
  const std::vector<DirectionMap<3, BlockNeighbors<3>>> expected_neighbors{
      {{Direction<3>::upper_xi(), {1, aligned}}},
      {{Direction<3>::lower_xi(), {0, aligned}},
       {Direction<3>::upper_xi(), {2, aligned}}},
      {{Direction<3>::lower_xi(), {1, aligned}}}};
  const std::vector<std::unordered_set<Direction<3>>> expected_externals{
      {Direction<3>::lower_xi()}, {}, {Direction<3>::upper_xi()}};

  test_domain_construction(
      domain, expected_neighbors, expected_externals,
      build_coord_maps(inner_radius, outer_radius, radial_partitioning,
                       radial_distribution));

  // A single grid-point pair is broadcast to every shell.
  const domain::creators::AngularCartoonSphere2D broadcast_creator{
      inner_radius,
      outer_radius,
      radial_partitioning,
      radial_distribution,
      std::array<size_t, 2>{{6, 9}},
      std::vector<size_t>{1, 1, 1},
      nullptr,
      create_boundary_condition(),
      create_boundary_condition()};
  CHECK(broadcast_creator.initial_extents() ==
        std::vector<std::array<size_t, 3>>{
            {{6, 9, 1}}, {{6, 9, 1}}, {{6, 9, 1}}});
}

void test_no_boundary_conditions() {
  INFO("AngularCartoonSphere2D without boundary conditions");
  const domain::creators::AngularCartoonSphere2D creator{
      1.0,
      3.0,
      {1.5},
      {Distribution::Linear, Distribution::Linear},
      std::array<size_t, 2>{{4, 5}},
      std::vector<size_t>{0, 0}};
  CHECK(creator.external_boundary_conditions().empty());
  const auto domain =
      TestHelpers::domain::creators::test_domain_creator(creator, false);
  check_topologies(domain);
}

void test_errors() {
  INFO("AngularCartoonSphere2D error conditions");
  const Options::Context context{
      .top_level = false, .context = {}, .line = 1, .column = 1};
  const std::array<size_t, 2> grid_points{{4, 5}};

  CHECK_THROWS_WITH(
      domain::creators::AngularCartoonSphere2D(
          0.0, 2.0, {}, {Distribution::Linear}, grid_points,
          std::vector<size_t>{0}, nullptr, nullptr, nullptr, context),
      Catch::Matchers::ContainsSubstring("InnerRadius must be positive"));

  CHECK_THROWS_WITH(
      domain::creators::AngularCartoonSphere2D(
          2.0, 1.0, {}, {Distribution::Linear}, grid_points,
          std::vector<size_t>{0}, nullptr, nullptr, nullptr, context),
      Catch::Matchers::ContainsSubstring(
          "InnerRadius must be smaller than OuterRadius"));

  CHECK_THROWS_WITH(
      domain::creators::AngularCartoonSphere2D(
          1.0, 4.0, {2.0}, {Distribution::Linear, Distribution::Linear},
          grid_points, std::vector<size_t>{0}, nullptr, nullptr, nullptr,
          context),
      Catch::Matchers::ContainsSubstring(
          "InitialRefinementInR must have one entry per radial block"));

  CHECK_THROWS_WITH(
      domain::creators::AngularCartoonSphere2D(
          1.0, 4.0, {3.0, 2.0},
          {Distribution::Linear, Distribution::Linear, Distribution::Linear},
          grid_points, std::vector<size_t>{0, 0, 0}, nullptr, nullptr, nullptr,
          context),
      Catch::Matchers::ContainsSubstring(
          "You must specify RadialPartitioning in ascending order"));

  CHECK_THROWS_WITH(
      domain::creators::AngularCartoonSphere2D(
          1.0, 4.0, {2.0, 2.0},
          {Distribution::Linear, Distribution::Linear, Distribution::Linear},
          grid_points, std::vector<size_t>{0, 0, 0}, nullptr, nullptr, nullptr,
          context),
      Catch::Matchers::ContainsSubstring(
          "RadialPartitioning contains duplicate element"));

  CHECK_THROWS_WITH(
      domain::creators::AngularCartoonSphere2D(
          1.0, 4.0, {0.5}, {Distribution::Linear, Distribution::Linear},
          grid_points, std::vector<size_t>{0, 0}, nullptr, nullptr, nullptr,
          context),
      Catch::Matchers::ContainsSubstring(
          "Radial partitions must be larger than InnerRadius"));

  CHECK_THROWS_WITH(
      domain::creators::AngularCartoonSphere2D(
          1.0, 4.0, {5.0}, {Distribution::Linear, Distribution::Linear},
          grid_points, std::vector<size_t>{0, 0}, nullptr, nullptr, nullptr,
          context),
      Catch::Matchers::ContainsSubstring(
          "Radial partitions must be smaller than OuterRadius"));

  CHECK_THROWS_WITH(
      domain::creators::AngularCartoonSphere2D(
          1.0, 4.0, {2.0}, {Distribution::Linear}, grid_points,
          std::vector<size_t>{0, 0}, nullptr, nullptr, nullptr, context),
      Catch::Matchers::ContainsSubstring(
          "RadialDistribution must have one entry per radial block"));

  CHECK_THROWS_WITH(
      domain::creators::AngularCartoonSphere2D(
          1.0, 4.0, {2.0}, {Distribution::Linear, Distribution::Linear},
          std::vector<std::array<size_t, 2>>{{{4, 5}}},
          std::vector<size_t>{0, 0}, nullptr, nullptr, nullptr, context),
      Catch::Matchers::ContainsSubstring(
          "InitialGridPoints must have one entry per radial block"));

  // An even number of azimuthal grid points is fine: unlike the full Fourier
  // basis, HalfFourier has no odd-extent requirement.
  CHECK_NOTHROW(domain::creators::AngularCartoonSphere2D(
      1.0, 2.0, {}, {Distribution::Linear}, std::array<size_t, 2>{{4, 6}},
      std::vector<size_t>{0}, nullptr, nullptr, nullptr, context));

  CHECK_THROWS_WITH(domain::creators::AngularCartoonSphere2D(
                        1.0, 2.0, {}, {Distribution::Linear}, grid_points,
                        std::vector<size_t>{0}, nullptr,
                        create_boundary_condition(), nullptr, context),
                    Catch::Matchers::ContainsSubstring(
                        "Either both InnerBoundary and OuterBoundary"));
  CHECK_THROWS_WITH(domain::creators::AngularCartoonSphere2D(
                        1.0, 2.0, {}, {Distribution::Linear}, grid_points,
                        std::vector<size_t>{0}, nullptr, nullptr,
                        create_boundary_condition(), context),
                    Catch::Matchers::ContainsSubstring(
                        "Either both InnerBoundary and OuterBoundary"));

  CHECK_THROWS_WITH(
      domain::creators::AngularCartoonSphere2D(
          1.0, 2.0, {}, {Distribution::Linear}, grid_points,
          std::vector<size_t>{0}, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          create_boundary_condition(), context),
      Catch::Matchers::ContainsSubstring(
          "None boundary condition is not supported"));
  CHECK_THROWS_WITH(
      domain::creators::AngularCartoonSphere2D(
          1.0, 2.0, {}, {Distribution::Linear}, grid_points,
          std::vector<size_t>{0}, nullptr, create_boundary_condition(),
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          context),
      Catch::Matchers::ContainsSubstring(
          "None boundary condition is not supported"));

  CHECK_THROWS_WITH(
      domain::creators::AngularCartoonSphere2D(
          1.0, 2.0, {}, {Distribution::Linear}, grid_points,
          std::vector<size_t>{0}, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          create_boundary_condition(), context),
      Catch::Matchers::ContainsSubstring(
          "Cannot have periodic boundary conditions on a 2D sphere"));
  CHECK_THROWS_WITH(
      domain::creators::AngularCartoonSphere2D(
          1.0, 2.0, {}, {Distribution::Linear}, grid_points,
          std::vector<size_t>{0}, nullptr, create_boundary_condition(),
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          context),
      Catch::Matchers::ContainsSubstring(
          "Cannot have periodic boundary conditions on a 2D sphere"));
}

void test_factory() {
  INFO("AngularCartoonSphere2D factory");
  const auto creator = TestHelpers::test_option_tag<
      domain::OptionTags::DomainCreator<3>,
      TestHelpers::domain::BoundaryConditions::
          MetavariablesWithBoundaryConditions<
              3, domain::creators::AngularCartoonSphere2D>>(
      "AngularCartoonSphere2D:\n"
      "  InnerRadius: 1.0\n"
      "  OuterRadius: 4.0\n"
      "  RadialPartitioning: [2.0]\n"
      "  RadialDistribution: [Linear, Logarithmic]\n"
      "  InitialGridPoints: [4, 6]\n"
      "  InitialRefinementInR: [1, 2]\n"
      "  TimeDependence: None\n"
      "  BoundaryConditions:\n"
      "    InnerBoundary:\n"
      "      TestBoundaryCondition:\n"
      "        Direction: lower-xi\n"
      "        BlockId: 0\n"
      "    OuterBoundary:\n"
      "      TestBoundaryCondition:\n"
      "        Direction: lower-xi\n"
      "        BlockId: 0\n");

  const auto* const cartoon_creator =
      dynamic_cast<const domain::creators::AngularCartoonSphere2D*>(
          creator.get());
  REQUIRE(cartoon_creator != nullptr);
  CHECK(cartoon_creator->block_names() ==
        std::vector<std::string>{"Shell0", "Shell1"});
  CHECK(cartoon_creator->initial_extents() ==
        std::vector<std::array<size_t, 3>>{{{4, 6, 1}}, {{4, 6, 1}}});
  CHECK(cartoon_creator->initial_refinement_levels() ==
        std::vector<std::array<size_t, 3>>{{{1, 0, 0}}, {{2, 0, 0}}});
  check_topologies(cartoon_creator->create_domain());
}

SPECTRE_TEST_CASE("Unit.Domain.Creators.AngularCartoonSphere2D",
                  "[Domain][Unit]") {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  test_half_plane_geometry();
  test_single_block();
  test_multiple_shells();
  test_no_boundary_conditions();
  test_errors();
  test_factory();
}
}  // namespace
