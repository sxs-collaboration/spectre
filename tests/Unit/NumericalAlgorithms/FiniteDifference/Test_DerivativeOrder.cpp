// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "Framework/TestCreation.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <fd::DerivativeOrder DerivOrder>
void test_construct_from_options(const std::string& expected_output) {
  CHECK(get_output(DerivOrder) == expected_output);
  const auto created =
      TestHelpers::test_creation<fd::DerivativeOrder>(get_output(DerivOrder));
  CHECK(created == DerivOrder);
}
}  // namespace


SPECTRE_TEST_CASE("Unit.FiniteDifference.DerivativeOrder",
                  "[Unit][NumericalAlgorithms]") {
  test_construct_from_options<fd::DerivativeOrder::OneHigherThanRecons>(
      "OneHigherThanRecons");
  test_construct_from_options<
      fd::DerivativeOrder::OneHigherThanReconsButFiveToFour>(
      "OneHigherThanReconsButFiveToFour");
  test_construct_from_options<fd::DerivativeOrder::Two>("2");
  test_construct_from_options<fd::DerivativeOrder::Four>("4");
  test_construct_from_options<fd::DerivativeOrder::Six>("6");
  test_construct_from_options<fd::DerivativeOrder::Eight>("8");
  test_construct_from_options<fd::DerivativeOrder::Ten>("10");
}
