// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.DG.Initialization.QuadratureTag",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<evolution::dg::Tags::Quadrature>(
      "Quadrature");
  CHECK(TestHelpers::test_creation<Spectral::Quadrature,
                                   evolution::dg::OptionTags::Quadrature>(
            "Gauss") == Spectral::Quadrature::Gauss);
  CHECK(TestHelpers::test_creation<Spectral::Quadrature,
                                   evolution::dg::OptionTags::Quadrature>(
            "GaussLobatto") == Spectral::Quadrature::GaussLobatto);
}
