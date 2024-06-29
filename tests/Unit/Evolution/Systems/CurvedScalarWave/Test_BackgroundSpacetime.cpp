// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/CurvedScalarWave/BackgroundSpacetime.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphericalKerrSchild.hpp"

namespace {
template <typename BackgroundType>
void test_option_parsing(const BackgroundType& background,
                         const std::string& option_string) {
  TestHelpers::db::test_simple_tag<
      CurvedScalarWave::Tags::BackgroundSpacetime<BackgroundType>>(
      "BackgroundSpacetime");
  CHECK(TestHelpers::test_option_tag<
            CurvedScalarWave::OptionTags::BackgroundSpacetime<BackgroundType>>(
            option_string) == background);
}
SPECTRE_TEST_CASE("Unit.Evolution.Systems.CurvedScalarWave.BackgroundSpacetime",
                  "[Unit][Evolution][Options]") {
  const auto minkowski_background_1d = gr::Solutions::Minkowski<1>();
  test_option_parsing(minkowski_background_1d, "");

  const auto minkowski_background_2d = gr::Solutions::Minkowski<2>();
  test_option_parsing(minkowski_background_2d, "");

  const auto minkowski_background_3d = gr::Solutions::Minkowski<3>();
  test_option_parsing(minkowski_background_3d, "");

  const auto ks_background =
      gr::Solutions::KerrSchild(0.5, {{0.1, 0.2, 0.3}}, {{1.0, 3.0, 2.0}});
  const std::string& ks_option_string =
      "Mass: 0.5\nSpin: [0.1,0.2,0.3]\nCenter: [1.0,3.0,2.0]\nVelocity: "
      "[0.0,0.0,0.0]";
  test_option_parsing(ks_background, ks_option_string);

  const auto spherical_ks_background = gr::Solutions::SphericalKerrSchild(
      0.5, {{0.1, 0.2, 0.3}}, {{1.0, 3.0, 2.0}});
  const std::string& spherical_ks_option_string =
      "Mass: 0.5\nSpin: [0.1,0.2,0.3]\nCenter: [1.0,3.0,2.0]";
  test_option_parsing(spherical_ks_background, spherical_ks_option_string);
}
}  // namespace
