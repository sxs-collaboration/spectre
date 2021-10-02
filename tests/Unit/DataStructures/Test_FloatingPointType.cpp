// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/FloatingPointType.hpp"
#include "Framework/TestCreation.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <FloatingPointType FloatType>
void test_construct_from_options() {
  const auto created =
      TestHelpers::test_creation<FloatingPointType>(get_output(FloatType));
  CHECK(created == FloatType);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.FloatingPointType",
                  "[Unit][DataStructures]") {
  CHECK(get_output(FloatingPointType::Float) == "Float");
  CHECK(get_output(FloatingPointType::Double) == "Double");

  test_construct_from_options<FloatingPointType::Float>();
  test_construct_from_options<FloatingPointType::Double>();
}
