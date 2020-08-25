// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Utilities/TypeTraits.hpp"

namespace domain {

namespace {
template <size_t Dim>
void test() {
  TestHelpers::db::test_simple_tag<Tags::InitialFunctionsOfTime<Dim>>(
      "InitialFunctionsOfTime");
  TestHelpers::db::test_simple_tag<Tags::FunctionsOfTime>("FunctionsOfTime");
  static_assert(
      std::is_same_v<typename Tags::FunctionsOfTime::type,
                     typename Tags::InitialFunctionsOfTime<Dim>::type>,
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
