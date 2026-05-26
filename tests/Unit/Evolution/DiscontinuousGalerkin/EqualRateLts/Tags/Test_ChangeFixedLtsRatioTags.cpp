// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/DiscontinuousGalerkin/EqualRateLts/Tags/ChangeFixedLtsRatioTags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
SPECTRE_TEST_CASE("Unit.Evolution.DG.EqualRateLts.Tags.ChangeFixedLtsRatioTags",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<
      evolution::dg::Tags::ChangeFixedLtsRatio::NumberOfExpectedMessages>(
      "NumberOfExpectedMessages");
  TestHelpers::db::test_simple_tag<
      evolution::dg::Tags::ChangeFixedLtsRatio::NewStepSize>("NewStepSize");
}
}  // namespace
