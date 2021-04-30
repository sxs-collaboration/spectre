// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/Formulation.hpp"

SPECTRE_TEST_CASE("Unit.DiscontinuousGalerkin.Tags.Formulation",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<dg::Tags::Formulation>("Formulation");
  CHECK(TestHelpers::test_option_tag<dg::OptionTags::Formulation>(
            "StrongInertial") == dg::Formulation::StrongInertial);
  CHECK(TestHelpers::test_option_tag<dg::OptionTags::Formulation>(
            "WeakInertial") == dg::Formulation::WeakInertial);
}
