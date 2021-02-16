// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"  // IWYU pragma: keep
#include "Domain/Block.hpp"                       // IWYU pragma: keep
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Frustum.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/FrustalCloak.hpp"
#include "Domain/Domain.hpp"
#include "Domain/OptionTags.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace {
using BoundaryCondVector = std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>;

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_boundary_condition() {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::upper_zeta(), 50);
}

std::string boundary_conditions_string() {
  return {
      "  BoundaryCondition:\n"
      "    TestBoundaryCondition:\n"
      "      Direction: upper-zeta\n"
      "      BlockId: 50\n"};
}

auto create_boundary_conditions() {
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions_all_blocks{10};
  const auto boundary_condition = create_boundary_condition();
  for (auto& block : boundary_conditions_all_blocks) {
    block[Direction<3>::lower_zeta()] = boundary_condition->get_clone();
    block[Direction<3>::upper_zeta()] = boundary_condition->get_clone();
  }
  return boundary_conditions_all_blocks;
}

void test_frustal_cloak_construction(
    const domain::creators::FrustalCloak& frustal_cloak,
    const BoundaryCondVector& expected_external_boundary_conditions) {
  Parallel::register_classes_in_list<
      typename domain::creators::FrustalCloak::maps_list>();

  const auto test_impl = [&expected_external_boundary_conditions,
                          &frustal_cloak](const auto& domain) {
    test_initial_domain(domain, frustal_cloak.initial_refinement_levels());
    test_physical_separation(frustal_cloak.create_domain().blocks());

    for (size_t block_id = 0;
         block_id < expected_external_boundary_conditions.size(); ++block_id) {
      const auto& block = domain.blocks()[block_id];
      REQUIRE(block.external_boundaries().size() ==
              expected_external_boundary_conditions[block_id].size());
      for (const auto& [direction, expected_bc_ptr] :
           expected_external_boundary_conditions[block_id]) {
        REQUIRE(block.external_boundary_conditions().count(direction) == 1);
        REQUIRE(block.external_boundary_conditions().at(direction) != nullptr);
        const auto& bc =
            dynamic_cast<const TestHelpers::domain::BoundaryConditions::
                             TestBoundaryCondition<3>&>(
                *block.external_boundary_conditions().at(direction));
        const auto& expected_bc =
            dynamic_cast<const TestHelpers::domain::BoundaryConditions::
                             TestBoundaryCondition<3>&>(*expected_bc_ptr);
        CHECK(bc.direction() == expected_bc.direction());
        CHECK(bc.block_id() == expected_bc.block_id());
      }
    }
  };
  test_impl(frustal_cloak.create_domain());
  test_impl(serialize_and_deserialize(frustal_cloak.create_domain()));
}

void test_factory() {
  for (const bool with_boundary_conditions : {true, false}) {
    const std::string opt_string{
        "FrustalCloak:\n"
        "  InitialRefinement: 3\n"
        "  InitialGridPoints: [2,3]\n"
        "  UseEquiangularMap: true\n"
        "  ProjectionFactor: 0.3\n"
        "  LengthInnerCube: 15.5\n"
        "  LengthOuterCube: 42.4\n"
        "  OriginPreimage: [0.2,0.3,-0.1]\n" +
        std::string{with_boundary_conditions ? boundary_conditions_string()
                                             : ""}};
    const auto frustal_cloak = [&opt_string, &with_boundary_conditions]() {
      if (with_boundary_conditions) {
        return TestHelpers::test_factory_creation<
            DomainCreator<3>, domain::OptionTags::DomainCreator<3>,
            TestHelpers::domain::BoundaryConditions::
                MetavariablesWithBoundaryConditions<3>>(opt_string);
      } else {
        return TestHelpers::test_factory_creation<
            DomainCreator<3>, domain::OptionTags::DomainCreator<3>,
            TestHelpers::domain::BoundaryConditions::
                MetavariablesWithoutBoundaryConditions<3>>(opt_string);
      }
    }();
    test_frustal_cloak_construction(
        dynamic_cast<const domain::creators::FrustalCloak&>(*frustal_cloak),
        with_boundary_conditions ? create_boundary_conditions()
                                 : BoundaryCondVector{});
  }
}

void test_connectivity() {
  const size_t refinement = 1;
  const std::array<size_t, 2> grid_points = {{6, 5}};
  const double projective_scale_factor = 0.3;
  const double length_inner_cube = 15.5;
  const double length_outer_cube = 42.4;
  const std::array<double, 3> origin_preimage = {{1.3, 0.2, -3.1}};

  for (const bool with_boundary_conditions : {true, false}) {
    for (const bool use_equiangular_map : {true, false}) {
      const domain::creators::FrustalCloak frustal_cloak{
          refinement,
          grid_points,
          use_equiangular_map,
          projective_scale_factor,
          length_inner_cube,
          length_outer_cube,
          origin_preimage,
          with_boundary_conditions ? create_boundary_condition() : nullptr};
      test_frustal_cloak_construction(
          frustal_cloak, with_boundary_conditions ? create_boundary_conditions()
                                                  : BoundaryCondVector{});
    }
  }

  CHECK_THROWS_WITH(
      domain::creators::FrustalCloak(
          refinement, grid_points, true, projective_scale_factor,
          length_inner_cube, length_outer_cube, origin_preimage,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Cannot have periodic boundary conditions with a frustal cloak"));
  CHECK_THROWS_WITH(
      domain::creators::FrustalCloak(
          refinement, grid_points, true, projective_scale_factor,
          length_inner_cube, length_outer_cube, origin_preimage,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "None boundary condition is not supported. If you would like an "
          "outflow boundary condition, you must use that."));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.FrustalCloak",
                  "[Domain][Unit]") {
  TestHelpers::domain::BoundaryConditions::register_derived_with_charm();
  test_factory();
  test_connectivity();
}
