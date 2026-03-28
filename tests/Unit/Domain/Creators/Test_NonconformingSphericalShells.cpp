// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
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
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_boundary_condition(const bool outer) {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      outer ? Direction<3>::upper_xi() : Direction<3>::lower_zeta(), 50);
}

std::string option_string(
    const double inner_radius, const double interface_radius,
    const double outer_radius, const size_t radial_refinement,
    const size_t angular_refinement, const size_t radial_extents,
    const size_t spherical_harmonic_l, const size_t angular_extents,
    const bool with_boundary_conditions) {
  const std::string inner_bc_option = with_boundary_conditions
                                          ? "  InnerBoundaryCondition:\n"
                                            "    TestBoundaryCondition:\n"
                                            "      Direction: lower-xi\n"
                                            "      BlockId: 50\n"
                                          : "";
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
         std::to_string(outer_radius) + "\n" +
         "  InitialRadialRefinement: " + std::to_string(radial_refinement) +
         "\n"
         "  InitialAngularRefinementOfWedges: " +
         std::to_string(angular_refinement) +
         "\n"
         "  InitialNumberOfRadialGridPoints: " +
         std::to_string(radial_extents) +
         "\n"
         "  InitialSphericalHarmonicL: " +
         std::to_string(spherical_harmonic_l) +
         "\n"
         "  InitialNumberOfAngularGridPointsOfWedges: " +
         std::to_string(angular_extents) + "\n" + inner_bc_option +
         outer_bc_option;
}

void test_parse_errors() {
  INFO("NonconformingSphericalShells check throws");
  const double inner_radius = 1.9;
  const double interface_radius = 2.4;
  const double outer_radius = 2.9;
  const size_t radial_refinement = 0;
  const size_t angular_refinement = 1;
  const size_t radial_extents = 12;
  const size_t l = 9;
  const size_t angular_extents = 11;

  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, 0.5 * inner_radius, outer_radius, radial_refinement,
          angular_refinement, radial_extents, l, angular_extents, nullptr,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Inner radius must be smaller than interface radius"));

  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, 1.5 * outer_radius, outer_radius, radial_refinement,
          angular_refinement, radial_extents, l, angular_extents, nullptr,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Interface radius must be smaller than outer radius"));

  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, radial_refinement,
          angular_refinement, radial_extents, l, angular_extents,
          create_boundary_condition(false), nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Must specify either both inner and outer boundary conditions "
          "or neither."));
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, radial_refinement,
          angular_refinement, radial_extents, l, angular_extents,
          create_boundary_condition(false),
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Cannot have periodic boundary conditions with "
          "NonconformingSphericalShells"));
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, radial_refinement,
          angular_refinement, radial_extents, l, angular_extents,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          create_boundary_condition(true), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Cannot have periodic boundary conditions with "
          "NonconformingSphericalShells"));
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, radial_refinement,
          angular_refinement, radial_extents, l, angular_extents,
          create_boundary_condition(false),
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "None boundary condition is not supported. If you would like "
          "an outflow-type boundary condition, you must use that."));
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, radial_refinement,
          angular_refinement, radial_extents, l, angular_extents,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          create_boundary_condition(true), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "None boundary condition is not supported. If you would like "
          "an outflow-type boundary condition, you must use that."));
  CHECK_THROWS_WITH(
      [&]() {
        const auto invalid_shell =
            domain::creators::NonconformingSphericalShells(
                0.0, interface_radius, outer_radius, radial_refinement,
                angular_refinement, radial_extents, l, angular_extents, nullptr,
                nullptr, Options::Context{false, {}, 1, 1});
        invalid_shell.create_domain();
      }(),
      Catch::Matchers::ContainsSubstring(
          "The radius of the inner surface must be greater than zero."));
}

template <typename Generator>
void test_nonconforming_spherical_shells_construction(
    const gsl::not_null<Generator*> gen,
    const domain::creators::NonconformingSphericalShells& creator,
    const double inner_radius, const double interface_radius,
    const double outer_radius, const bool expect_boundary_conditions = true) {
  // check consistency of domain
  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      creator, expect_boundary_conditions);
  const auto& grid_anchors = creator.grid_anchors();
  CHECK(grid_anchors.size() == 1);
  CHECK(grid_anchors.count("Center") == 1);
  CHECK(grid_anchors.at("Center") ==
        tnsr::I<double, 3, Frame::Grid>{std::array{0.0, 0.0, 0.0}});

  const auto& blocks = domain.blocks();
  const auto block_names = creator.block_names();
  const size_t num_blocks = blocks.size();
  CAPTURE(num_blocks);
  const auto all_boundary_conditions = creator.external_boundary_conditions();
  const auto functions_of_time = creator.functions_of_time();

  // Check total number of external boundaries
  const size_t num_external_boundaries =
      alg::accumulate(blocks, 0_st, [](const size_t count, const auto& block) {
        return count + block.external_boundaries().size();
      });
  CHECK(num_external_boundaries == 7);

  // NOLINTNEXTLINE(misc-const-correctness)
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  for (size_t block_id = 0; block_id < num_blocks - 1; ++block_id) {
    CAPTURE(block_id);
    const auto& block = blocks[block_id];
    const ElementMap<3, Frame::Grid> grid_element_map{ElementId<3>{block_id},
                                                      block};
    const ElementMap<3, Frame::Inertial> inertial_element_map{
        ElementId<3>{block_id}, block};
    {
      INFO("Radius of random point on lower face of cubed shells");
      const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
          {{xi_distribution(*gen), xi_distribution(*gen), -1.0}}};
      auto x_inertial = inertial_element_map(x_logical);
      CHECK(get(magnitude(x_inertial)) == approx(inner_radius));
    }
    {
      INFO("Radius of random point on upper face of cubed shells");
      const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
          {{xi_distribution(*gen), xi_distribution(*gen), 1.0}}};
      auto x_inertial = inertial_element_map(x_logical);
      CHECK(get(magnitude(x_inertial)) == approx(interface_radius));
    }
    {
      INFO("External boundaries of cubed shells");
      const auto& external_boundaries = block.external_boundaries();
      CHECK(external_boundaries.size() == 1);
      CHECK(alg::found(external_boundaries, Direction<3>::lower_zeta()));
    }
    if (expect_boundary_conditions) {
      INFO("Boundary conditions of cubed shells");
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
  // NOLINTNEXTLINE(misc-const-correctness)
  std::uniform_real_distribution<> theta_distribution(0.0, M_PI);
  // NOLINTNEXTLINE(misc-const-correctness)
  std::uniform_real_distribution<> phi_distribution(0.0, 2.0 * M_PI);
  const auto& block = blocks[6];
  const ElementMap<3, Frame::Grid> grid_element_map{ElementId<3>{6}, block};
  const ElementMap<3, Frame::Inertial> inertial_element_map{ElementId<3>{6},
                                                            block};
  {
    INFO("Radius of random point on lower face of spherical shell");
    const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
        {{-1.0, theta_distribution(*gen), phi_distribution(*gen)}}};
    auto x_inertial = inertial_element_map(x_logical);
    CHECK(get(magnitude(x_inertial)) == approx(interface_radius));
  }
  {
    INFO("Radius of random point on upper face of spherical shell");
    const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
        {{1.0, theta_distribution(*gen), phi_distribution(*gen)}}};
    auto x_inertial = inertial_element_map(x_logical);
    CHECK(get(magnitude(x_inertial)) == approx(outer_radius));
  }
  {
    INFO("External boundaries of spherical shell");
    const auto& external_boundaries = block.external_boundaries();
    CHECK(external_boundaries.size() == 1);
    CHECK(alg::found(external_boundaries, Direction<3>::upper_xi()));
  }
  if (expect_boundary_conditions) {
    INFO("Boundary conditions of spherical shell");
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

template <typename Generator>
void test(const gsl::not_null<Generator*> gen) {
  const double inner_radius = 1.0;
  const double interface_radius = 1.5;
  const double outer_radius = 2.0;
  const size_t radial_refinement = 3;
  const size_t angular_refinement = 2;
  const size_t radial_extents = 5;
  const size_t l = 6;
  const size_t angular_extents = 7;
  for (const bool with_boundary_conditions : {true, false}) {
    CAPTURE(with_boundary_conditions);
    const domain::creators::NonconformingSphericalShells creator{
        inner_radius,
        interface_radius,
        outer_radius,
        radial_refinement,
        angular_refinement,
        radial_extents,
        l,
        angular_extents,
        with_boundary_conditions ? create_boundary_condition(false) : nullptr,
        with_boundary_conditions ? create_boundary_condition(true) : nullptr};
    test_nonconforming_spherical_shells_construction(
        gen, creator, inner_radius, interface_radius, outer_radius,
        with_boundary_conditions);
    TestHelpers::domain::creators::test_creation(
        option_string(inner_radius, interface_radius, outer_radius,
                      radial_refinement, angular_refinement, radial_extents, l,
                      angular_extents, with_boundary_conditions),
        creator, with_boundary_conditions);
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
