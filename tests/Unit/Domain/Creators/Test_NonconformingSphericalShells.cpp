// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/NonconformingSphericalShells.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Options/Context.hpp"
#include "Utilities/Gsl.hpp"

namespace {

using Excision = domain::creators::NonconformingSphericalShells::Excision;
using InnerCube = domain::creators::NonconformingSphericalShells::InnerCube;
using Distribution = domain::CoordinateMaps::Distribution;

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_inner_boundary_condition() {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::lower_zeta(), 50);
}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_outer_boundary_condition() {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::upper_xi(), 50);
}

std::string option_string(const double inner_radius,
                          const double interface_radius,
                          const double outer_radius,
                          const std::array<size_t, 2>& cube_refinement,
                          const size_t sh_refinement,
                          const std::array<size_t, 2>& cube_grid_points,
                          const std::array<size_t, 2>& sh_grid_points,
                          const bool with_boundary_conditions) {
  const std::string interior_option = with_boundary_conditions
                                          ? "  Interior:\n"
                                            "    ExciseWithBoundaryCondition:\n"
                                            "      TestBoundaryCondition:\n"
                                            "        Direction: lower-zeta\n"
                                            "        BlockId: 50\n"
                                          : "  Interior: Excise\n";
  const std::string outer_bc_option = with_boundary_conditions
                                          ? "  OuterBoundaryCondition:\n"
                                            "    TestBoundaryCondition:\n"
                                            "      Direction: upper-xi\n"
                                            "      BlockId: 50\n"
                                          : "";
  return "NonconformingSphericalShells:\n"
         "  InnerRadius: " +
         std::to_string(inner_radius) +
         "\n"
         "  InterfaceRadius: " +
         std::to_string(interface_radius) +
         "\n"
         "  OuterRadius: " +
         std::to_string(outer_radius) + "\n" + interior_option +
         "  InitialCubeRefinement: [" + std::to_string(cube_refinement[0]) +
         ", " + std::to_string(cube_refinement[1]) +
         "]\n"
         "  InitialSHRefinement: " +
         std::to_string(sh_refinement) +
         "\n"
         "  InitialCubeGridPoints: [" +
         std::to_string(cube_grid_points[0]) + ", " +
         std::to_string(cube_grid_points[1]) +
         "]\n"
         "  InitialSHGridPoints: [" +
         std::to_string(sh_grid_points[0]) + ", " +
         std::to_string(sh_grid_points[1]) +
         "]\n"
         "  RadialPartitioning: [[], []]\n"
         "  RadialDistribution: [[Linear], [Linear]]\n"
         "  UseEquiangularMap: false\n"
         "  TimeDependentMaps: None\n" +
         outer_bc_option;
}

void test_parse_errors() {
  INFO("NonconformingSphericalShells check throws");
  const double inner_radius = 1.9;
  const double interface_radius = 2.4;
  const double outer_radius = 2.9;
  const std::array<size_t, 2> cube_ref{1_st, 0_st};
  const size_t sh_ref = 0;
  const std::array<size_t, 2> cube_gp{3_st, 4_st};
  const std::array<size_t, 2> sh_gp{2_st, 3_st};
  const std::array<std::vector<double>, 2> radial_partitioning{};
  const std::array<std::vector<Distribution>, 2> radial_distribution{
      std::vector<Distribution>{Distribution::Linear},
      std::vector<Distribution>{Distribution::Linear}};

  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, 0.5 * inner_radius, outer_radius,
          Excision{create_inner_boundary_condition()}, cube_ref, sh_ref,
          cube_gp, sh_gp, radial_partitioning, radial_distribution, false,
          std::nullopt, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Inner radius must be smaller than interface radius"));

  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, 1.5 * outer_radius, outer_radius,
          Excision{create_inner_boundary_condition()}, cube_ref, sh_ref,
          cube_gp, sh_gp, radial_partitioning, radial_distribution, false,
          std::nullopt, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Interface radius must be smaller than outer radius"));

  // Inner BC only, no outer BC
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius,
          Excision{create_inner_boundary_condition()}, cube_ref, sh_ref,
          cube_gp, sh_gp, radial_partitioning, radial_distribution, false,
          std::nullopt, nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Must specify either both inner and outer boundary conditions "
          "or neither."));

  // Periodic inner BC
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius,
          Excision{std::make_unique<TestHelpers::domain::BoundaryConditions::
                                        TestPeriodicBoundaryCondition<3>>()},
          cube_ref, sh_ref, cube_gp, sh_gp, radial_partitioning,
          radial_distribution, false, std::nullopt,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Cannot have periodic boundary conditions with "
          "NonconformingSphericalShells"));

  // Periodic outer BC
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius,
          Excision{create_inner_boundary_condition()}, cube_ref, sh_ref,
          cube_gp, sh_gp, radial_partitioning, radial_distribution, false,
          std::nullopt,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Cannot have periodic boundary conditions with "
          "NonconformingSphericalShells"));

  // None outer BC
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius,
          Excision{create_inner_boundary_condition()}, cube_ref, sh_ref,
          cube_gp, sh_gp, radial_partitioning, radial_distribution, false,
          std::nullopt,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "None boundary condition is not supported for the outer boundary"));

  // None inner BC (excision)
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius,
          Excision{std::make_unique<TestHelpers::domain::BoundaryConditions::
                                        TestNoneBoundaryCondition<3>>()},
          cube_ref, sh_ref, cube_gp, sh_gp, radial_partitioning,
          radial_distribution, false, std::nullopt,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "None boundary condition for the inner boundary is not supported "
          "when the center is excised"));

  // InnerCube with mismatched angular/radial refinement
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, InnerCube{0.0},
          std::array<size_t, 2>{1_st, 2_st}, sh_ref, cube_gp, sh_gp,
          radial_partitioning, radial_distribution, false, std::nullopt,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "The inner cube has different refinement for angular and radial "
          "input."));

  // InnerCube with mismatched angular/radial grid points
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, InnerCube{0.0},
          std::array<size_t, 2>{1_st, 1_st}, sh_ref,
          std::array<size_t, 2>{3_st, 4_st}, sh_gp, radial_partitioning,
          radial_distribution, false, std::nullopt, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "The inner cube has a different number of grid points for angular "
          "and radial input."));
}

// Test the excision case: 6 wedge blocks + 1 SH shell block.
// Verifies radii at block faces, external boundaries, and boundary conditions.
template <typename Generator>
void test_excision_construction(
    const gsl::not_null<Generator*> gen,
    const domain::creators::NonconformingSphericalShells& creator,
    const double inner_radius, const double interface_radius,
    const double outer_radius, const bool expect_boundary_conditions) {
  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      creator, expect_boundary_conditions);
  const auto& grid_anchors = creator.grid_anchors();
  CHECK(grid_anchors.size() == 1);
  CHECK(grid_anchors.count("Center") == 1);
  CHECK(grid_anchors.at("Center") ==
        tnsr::I<double, 3, Frame::Grid>{std::array{0.0, 0.0, 0.0}});

  const auto& blocks = domain.blocks();
  const size_t num_blocks = blocks.size();
  CAPTURE(num_blocks);
  // 6 wedge blocks + 1 SH shell
  CHECK(num_blocks == 7);

  // Check block groups: "Wedges" and "InnerRegion" both contain the 6 wedge
  // blocks (no inner cube in excision case), "InnerShells" is gone.
  {
    const auto& groups = creator.block_groups();
    const std::unordered_set<std::string> wedge_names{
        "InnerShell0UpperZ", "InnerShell0LowerZ", "InnerShell0UpperY",
        "InnerShell0LowerY", "InnerShell0UpperX", "InnerShell0LowerX"};
    REQUIRE(groups.count("Wedges") == 1);
    CHECK(groups.at("Wedges") == wedge_names);
    REQUIRE(groups.count("InnerRegion") == 1);
    CHECK(groups.at("InnerRegion") == wedge_names);
    REQUIRE(groups.count("InnerShell0") == 1);
    CHECK(groups.at("InnerShell0") == wedge_names);
    REQUIRE(groups.count("OuterShells") == 1);
    CHECK(groups.at("OuterShells") ==
          std::unordered_set<std::string>{"OuterShell0"});
    CHECK(not groups.contains("InnerShells"));
    CHECK(not groups.contains("InnerCube"));
  }

  const auto all_boundary_conditions = creator.external_boundary_conditions();

  // 6 inner (excision) + 1 outer
  const size_t num_external_boundaries =
      alg::accumulate(blocks, 0_st, [](const size_t count, const auto& block) {
        return count + block.external_boundaries().size();
      });
  CHECK(num_external_boundaries == 7);

  // NOLINTNEXTLINE(misc-const-correctness)
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);

  // Check wedge blocks (0–5)
  for (size_t block_id = 0; block_id < 6; ++block_id) {
    CAPTURE(block_id);
    const auto& block = blocks[block_id];
    const ElementMap<3, Frame::Inertial> inertial_element_map{
        ElementId<3>{block_id}, block};
    {
      INFO("Radius of random point on lower face of wedge block");
      const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
          {{xi_distribution(*gen), xi_distribution(*gen), -1.0}}};
      const auto x_inertial = inertial_element_map(x_logical);
      CHECK(get(magnitude(x_inertial)) == approx(inner_radius));
    }
    {
      INFO("Radius of random point on upper face of wedge block");
      const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
          {{xi_distribution(*gen), xi_distribution(*gen), 1.0}}};
      const auto x_inertial = inertial_element_map(x_logical);
      CHECK(get(magnitude(x_inertial)) == approx(interface_radius));
    }
    {
      INFO("External boundaries of wedge block");
      const auto& external_boundaries = block.external_boundaries();
      CHECK(external_boundaries.size() == 1);
      CHECK(alg::found(external_boundaries, Direction<3>::lower_zeta()));
    }
    if (expect_boundary_conditions) {
      INFO("Boundary conditions of wedge block");
      const auto& boundary_conditions = all_boundary_conditions[block_id];
      for (const auto& direction : block.external_boundaries()) {
        CAPTURE(direction);
        const auto& boundary_condition =
            dynamic_cast<const TestHelpers::domain::BoundaryConditions::
                             TestBoundaryCondition<3>&>(
                *boundary_conditions.at(direction));
        CHECK(boundary_condition.direction() == direction);
      }
    }
  }

  // Check SH shell block (block 6)
  // NOLINTNEXTLINE(misc-const-correctness)
  std::uniform_real_distribution<> theta_distribution(0.0, M_PI);
  // NOLINTNEXTLINE(misc-const-correctness)
  std::uniform_real_distribution<> phi_distribution(0.0, 2.0 * M_PI);
  {
    const auto& block = blocks[6];
    const ElementMap<3, Frame::Inertial> inertial_element_map{ElementId<3>{6},
                                                              block};
    {
      INFO("Radius of random point on inner face of SH shell");
      const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
          {{-1.0, theta_distribution(*gen), phi_distribution(*gen)}}};
      const auto x_inertial = inertial_element_map(x_logical);
      CHECK(get(magnitude(x_inertial)) == approx(interface_radius));
    }
    {
      INFO("Radius of random point on outer face of SH shell");
      const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
          {{1.0, theta_distribution(*gen), phi_distribution(*gen)}}};
      const auto x_inertial = inertial_element_map(x_logical);
      CHECK(get(magnitude(x_inertial)) == approx(outer_radius));
    }
    {
      INFO("External boundaries of SH shell");
      const auto& external_boundaries = block.external_boundaries();
      CHECK(external_boundaries.size() == 1);
      CHECK(alg::found(external_boundaries, Direction<3>::upper_xi()));
    }
    if (expect_boundary_conditions) {
      INFO("Boundary conditions of SH shell");
      const auto& boundary_conditions = all_boundary_conditions[6];
      for (const auto& direction : block.external_boundaries()) {
        CAPTURE(direction);
        const auto& boundary_condition =
            dynamic_cast<const TestHelpers::domain::BoundaryConditions::
                             TestBoundaryCondition<3>&>(
                *boundary_conditions.at(direction));
        CHECK(boundary_condition.direction() == direction);
      }
    }
  }
}

// Test the filled-interior case: 6 wedge blocks + 1 inner cube + 1 SH shell.
template <typename Generator>
void test_fill_construction(
    const gsl::not_null<Generator*> gen,
    const domain::creators::NonconformingSphericalShells& creator,
    const double interface_radius, const double outer_radius) {
  const auto domain =
      TestHelpers::domain::creators::test_domain_creator(creator, false);
  const auto& blocks = domain.blocks();
  const size_t num_blocks = blocks.size();
  CAPTURE(num_blocks);
  // 6 wedge blocks + 1 inner cube + 1 SH shell
  CHECK(num_blocks == 8);

  // No excision sphere → only 1 outer external boundary
  const size_t num_external_boundaries =
      alg::accumulate(blocks, 0_st, [](const size_t count, const auto& block) {
        return count + block.external_boundaries().size();
      });
  CHECK(num_external_boundaries == 1);

  // Check block groups: "InnerRegion" includes wedges + inner cube block;
  // "Wedges" contains only the wedge blocks; "InnerShells" is gone.
  {
    const auto& groups = creator.block_groups();
    const std::unordered_set<std::string> wedge_names{
        "InnerShell0UpperZ", "InnerShell0LowerZ", "InnerShell0UpperY",
        "InnerShell0LowerY", "InnerShell0UpperX", "InnerShell0LowerX"};
    std::unordered_set<std::string> inner_region_names = wedge_names;
    inner_region_names.insert("InnerCube");
    REQUIRE(groups.count("Wedges") == 1);
    CHECK(groups.at("Wedges") == wedge_names);
    REQUIRE(groups.count("InnerRegion") == 1);
    CHECK(groups.at("InnerRegion") == inner_region_names);
    REQUIRE(groups.count("OuterShells") == 1);
    CHECK(groups.at("OuterShells") ==
          std::unordered_set<std::string>{"OuterShell0"});
    CHECK(not groups.contains("InnerShells"));
    CHECK(not groups.contains("InnerCube"));
  }

  // Check SH shell block (block 7 in the fill case)
  // NOLINTNEXTLINE(misc-const-correctness)
  std::uniform_real_distribution<> theta_distribution(0.0, M_PI);
  // NOLINTNEXTLINE(misc-const-correctness)
  std::uniform_real_distribution<> phi_distribution(0.0, 2.0 * M_PI);
  {
    const auto& block = blocks[7];
    const ElementMap<3, Frame::Inertial> inertial_element_map{ElementId<3>{7},
                                                              block};
    {
      INFO("Radius of random point on inner face of SH shell (fill case)");
      const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
          {{-1.0, theta_distribution(*gen), phi_distribution(*gen)}}};
      const auto x_inertial = inertial_element_map(x_logical);
      CHECK(get(magnitude(x_inertial)) == approx(interface_radius));
    }
    {
      INFO("Radius of random point on outer face of SH shell (fill case)");
      const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
          {{1.0, theta_distribution(*gen), phi_distribution(*gen)}}};
      const auto x_inertial = inertial_element_map(x_logical);
      CHECK(get(magnitude(x_inertial)) == approx(outer_radius));
    }
    {
      INFO("External boundaries of SH shell (fill case)");
      const auto& external_boundaries = block.external_boundaries();
      CHECK(external_boundaries.size() == 1);
      CHECK(alg::found(external_boundaries, Direction<3>::upper_xi()));
    }
  }
}

template <typename Generator>
void test(const gsl::not_null<Generator*> gen) {
  const double inner_radius = 1.0;
  const double interface_radius = 1.5;
  const double outer_radius = 2.0;
  const std::array<size_t, 2> cube_ref{2_st, 1_st};
  const size_t sh_ref = 0;
  const std::array<size_t, 2> cube_gp{3_st, 4_st};
  const std::array<size_t, 2> sh_gp{4_st, 3_st};
  const std::array<std::vector<double>, 2> radial_partitioning{};
  const std::array<std::vector<Distribution>, 2> radial_distribution{
      std::vector<Distribution>{Distribution::Linear},
      std::vector<Distribution>{Distribution::Linear}};

  // Test excision case with and without boundary conditions
  for (const bool with_boundary_conditions : {true, false}) {
    CAPTURE(with_boundary_conditions);
    const domain::creators::NonconformingSphericalShells creator{
        inner_radius,
        interface_radius,
        outer_radius,
        Excision{with_boundary_conditions ? create_inner_boundary_condition()
                                          : nullptr},
        cube_ref,
        sh_ref,
        cube_gp,
        sh_gp,
        radial_partitioning,
        radial_distribution,
        false,
        std::nullopt,
        with_boundary_conditions ? create_outer_boundary_condition() : nullptr};
    test_excision_construction(gen, creator, inner_radius, interface_radius,
                               outer_radius, with_boundary_conditions);
    TestHelpers::domain::creators::test_creation(
        option_string(inner_radius, interface_radius, outer_radius, cube_ref,
                      sh_ref, cube_gp, sh_gp, with_boundary_conditions),
        creator, with_boundary_conditions);
  }

  // Test filled-interior case (no boundary conditions).
  // The inner cube requires equal angular/radial refinement and grid points.
  {
    const std::array<size_t, 2> fill_cube_ref{2_st, 2_st};
    const std::array<size_t, 2> fill_cube_gp{3_st, 3_st};
    const domain::creators::NonconformingSphericalShells creator{
        inner_radius,
        interface_radius,
        outer_radius,
        InnerCube{0.0},
        fill_cube_ref,
        sh_ref,
        fill_cube_gp,
        sh_gp,
        radial_partitioning,
        std::array<std::vector<Distribution>, 2>{
            std::vector<Distribution>{Distribution::Linear},
            std::vector<Distribution>{Distribution::Linear}},
        false,
        std::nullopt,
        nullptr};
    test_fill_construction(gen, creator, interface_radius, outer_radius);
  }
}
}  // namespace

// [[TimeOut, 15]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.NonconformingSphericalShells",
                  "[Domain][Unit]") {
  MAKE_GENERATOR(gen);
  domain::creators::time_dependence::register_derived_with_charm();
  test_parse_errors();
  test(make_not_null(&gen));
}
