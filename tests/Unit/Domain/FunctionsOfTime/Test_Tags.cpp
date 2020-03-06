// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"

namespace domain {

namespace {
template <size_t Dim>
void test() {
  TestHelpers::db::test_simple_tag<Tags::InitialFunctionsOfTime<Dim>>(
      "InitialFunctionsOfTime");
  TestHelpers::db::test_simple_tag<Tags::FunctionsOfTime>("FunctionsOfTime");
  static_assert(
      cpp17::is_same_v<db::item_type<Tags::FunctionsOfTime>,
                       db::item_type<Tags::InitialFunctionsOfTime<Dim>>>,
      "FunctionsOfTime and InitialFunctionsOfTime tags must have the same "
      "types");
}

SPECTRE_TEST_CASE("Unit.Domain.FunctionsOfTime.Tags",
                  "[Domain][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
}  // namespace domain
