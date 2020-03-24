// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/Tags.hpp"

namespace {
struct SomeType {};
}  // namespace

SPECTRE_TEST_CASE("Unit.LinearOperators.Tags", "[Unit][NumericalAlgorithms]") {
  TestHelpers::db::test_simple_tag<Filters::Tags::Filter<SomeType>>("Filter");
}
