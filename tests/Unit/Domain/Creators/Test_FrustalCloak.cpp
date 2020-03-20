// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"  // IWYU pragma: keep
#include "Domain/Block.hpp"                       // IWYU pragma: keep
#include "Domain/BlockNeighbor.hpp"               // IWYU pragma: keep
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Frustum.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/FrustalCloak.hpp"
#include "Domain/Domain.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace {
void test_frustal_cloak_construction(
    const domain::creators::FrustalCloak& frustal_cloak) {
  const auto domain = frustal_cloak.create_domain();
  test_initial_domain(domain, frustal_cloak.initial_refinement_levels());
  test_physical_separation(frustal_cloak.create_domain().blocks());

  Parallel::register_classes_in_list<
      typename domain::creators::FrustalCloak::maps_list>();
  test_serialization(domain);
}

void test_factory() {
  const auto frustal_cloak =
      TestHelpers::test_factory_creation<DomainCreator<3>>(
          "FrustalCloak:\n"
          "  InitialRefinement: 3\n"
          "  InitialGridPoints: [2,3]\n"
          "  UseEquiangularMap: true\n"
          "  ProjectionFactor: 0.3\n"
          "  LengthInnerCube: 15.5\n"
          "  LengthOuterCube: 42.4\n"
          "  OriginPreimage: [0.2,0.3,-0.1]");
  test_frustal_cloak_construction(
      dynamic_cast<const domain::creators::FrustalCloak&>(*frustal_cloak));
}

void test_connectivity() {
  const size_t refinement = 1;
  const std::array<size_t, 2> grid_points = {{6, 5}};
  const double projective_scale_factor = 0.3;
  const double length_inner_cube = 15.5;
  const double length_outer_cube = 42.4;
  const std::array<double, 3> origin_preimage = {{1.3, 0.2, -3.1}};

  for (const bool use_equiangular_map : {true, false}) {
    const domain::creators::FrustalCloak frustal_cloak{
        refinement,          grid_points,
        use_equiangular_map, projective_scale_factor,
        length_inner_cube,   length_outer_cube,
        origin_preimage};
    test_frustal_cloak_construction(frustal_cloak);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.FrustalCloak",
                  "[Domain][Unit]") {
  test_factory();
  test_connectivity();
}
