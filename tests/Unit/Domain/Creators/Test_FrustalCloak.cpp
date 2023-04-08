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
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace {
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
        return TestHelpers::test_option_tag<
            domain::OptionTags::DomainCreator<3>,
            TestHelpers::domain::BoundaryConditions::
                MetavariablesWithBoundaryConditions<
                    3, domain::creators::FrustalCloak>>(opt_string);
      } else {
        return TestHelpers::test_option_tag<
            domain::OptionTags::DomainCreator<3>,
            TestHelpers::domain::BoundaryConditions::
                MetavariablesWithoutBoundaryConditions<
                    3, domain::creators::FrustalCloak>>(opt_string);
      }
    }();
    TestHelpers::domain::creators::test_domain_creator(
        *frustal_cloak, with_boundary_conditions);
  }
}

void test_connectivity() {
  const size_t refinement = 1;
  const std::array<size_t, 2> grid_points = {{6, 5}};
  const double projective_scale_factor = 0.3;
  const double length_inner_cube = 15.5;
  const double length_outer_cube = 42.4;
  const std::array<double, 3> origin_preimage = {{1.3, 0.2, -3.1}};

  for (const auto& [with_boundary_conditions, use_equiangular_map] :
       cartesian_product(make_array(true, false), make_array(true, false))) {
    CAPTURE(with_boundary_conditions);
    CAPTURE(use_equiangular_map);
    const domain::creators::FrustalCloak frustal_cloak{
        refinement,
        grid_points,
        use_equiangular_map,
        projective_scale_factor,
        length_inner_cube,
        length_outer_cube,
        origin_preimage,
        with_boundary_conditions ? create_boundary_condition() : nullptr};
    TestHelpers::domain::creators::test_domain_creator(
        frustal_cloak, with_boundary_conditions);
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
          "outflow-type boundary condition, you must use that."));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.FrustalCloak",
                  "[Domain][Unit]") {
  TestHelpers::domain::BoundaryConditions::register_derived_with_charm();
  test_factory();
  test_connectivity();
}
