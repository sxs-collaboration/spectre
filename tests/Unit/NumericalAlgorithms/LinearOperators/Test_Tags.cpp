// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "NumericalAlgorithms/LinearOperators/Tags.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"

namespace {
struct SomeType {};
}  // namespace

SPECTRE_TEST_CASE("Unit.LinearOperators.Tags", "[Unit][NumericalAlgorithms]") {
  TestHelpers::db::test_simple_tag<Filters::Tags::Filter<SomeType>>("Filter");
}
