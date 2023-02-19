// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
void test_excision_sphere_tag() {
  const std::unique_ptr<DomainCreator<3>> shell =
      std::make_unique<domain::creators::Shell>(
          1., 2., 2, std::array<size_t, 2>{{4, 4}}, true);
  const auto shell_excision_sphere =
      shell->create_domain().excision_spheres().at("ExcisionSphere");

  CHECK(
      CurvedScalarWave::Worldtube::Tags::ExcisionSphere<3>::create_from_options(
          shell, "ExcisionSphere") == shell_excision_sphere);
  CHECK_THROWS_WITH(
      CurvedScalarWave::Worldtube::Tags::ExcisionSphere<3>::create_from_options(
          shell, "OtherExcisionSphere"),
      Catch::Matchers::Contains(
          "Specified excision sphere 'OtherExcisionSphere' not available. "
          "Available excision spheres are: (ExcisionSphere)"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.CurvedScalarWave.Worldtube.Tags",
                  "[Unit][Evolution]") {
  CHECK(TestHelpers::test_option_tag<
            CurvedScalarWave::Worldtube::OptionTags::ExcisionSphere>(
            "Worldtube") == "Worldtube");
  TestHelpers::db::test_simple_tag<
      CurvedScalarWave::Worldtube::Tags::ExcisionSphere<3>>("ExcisionSphere");
  TestHelpers::db::test_simple_tag<
      CurvedScalarWave::Worldtube::Tags::ElementFacesGridCoordinates<3>>(
      "ElementFacesGridCoordinates");
  test_excision_sphere_tag();
}
