// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/CurvedScalarWave/PsiSquared.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.CurvedScalarWave.PsiSquared",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<CurvedScalarWave::Tags::PsiSquared>(
      "PsiSquared");
  TestHelpers::db::test_compute_tag<CurvedScalarWave::Tags::PsiSquaredCompute>(
      "PsiSquared");
}
