// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Initialization/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Initialization.Tags",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<Initialization::Tags::InitialTime>(
      "InitialTime");
  TestHelpers::db::test_simple_tag<Initialization::Tags::InitialTimeDelta>(
      "InitialTimeDelta");
  TestHelpers::db::test_simple_tag<Initialization::Tags::InitialSlabSize<true>>(
      "InitialSlabSize");
  TestHelpers::db::test_simple_tag<
      Initialization::Tags::InitialSlabSize<false>>("InitialSlabSize");
}
