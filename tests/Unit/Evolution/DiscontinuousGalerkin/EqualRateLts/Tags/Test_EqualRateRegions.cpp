// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/DiscontinuousGalerkin/EqualRateLts/Tags/EqualRateRegions.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
SPECTRE_TEST_CASE("Unit.Evolution.DG.EqualRateLts.Tags.EqualRateRegions",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<
      evolution::dg::Tags::ConcreteEqualRateRegions<1, tmpl::list<>>>(
      "ConcreteEqualRateRegions");
  TestHelpers::db::test_simple_tag<evolution::dg::Tags::EqualRateRegions<1>>(
      "EqualRateRegions");
  TestHelpers::db::test_reference_tag<
      evolution::dg::Tags::EqualRateRegionsRef<1, tmpl::list<>>>(
      "EqualRateRegions");
}
}  // namespace
