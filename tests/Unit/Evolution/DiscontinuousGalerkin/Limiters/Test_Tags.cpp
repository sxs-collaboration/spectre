// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
struct SomeType {};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DiscontinuousGalerkin.Limiters.Tags",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<Tags::Limiter<SomeType>>("Limiter");
}
