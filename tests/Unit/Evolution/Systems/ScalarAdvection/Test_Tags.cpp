// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
template <size_t Dim>
void test_tags() {
  TestHelpers::db::test_simple_tag<ScalarAdvection::Tags::U>("U");
  TestHelpers::db::test_simple_tag<ScalarAdvection::Tags::VelocityField<Dim>>(
      "VelocityField");
  TestHelpers::db::test_simple_tag<
      ScalarAdvection::Tags::LargestCharacteristicSpeed>(
      "LargestCharacteristicSpeed");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarAdvection.Tags",
                  "[Unit][Evolution]") {
  test_tags<1>();
  test_tags<2>();
}
