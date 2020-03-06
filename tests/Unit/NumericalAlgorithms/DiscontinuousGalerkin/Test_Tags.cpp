// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"

namespace {
struct SomeType {};

struct SomeTag : db::SimpleTag {
  using type = int;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.DG.Tags", "[Unit][NumericalAlgorithms]") {
  TestHelpers::db::test_simple_tag<
      Tags::SimpleMortarData<SomeType, SomeType, SomeType>>("SimpleMortarData");
  TestHelpers::db::test_prefix_tag<Tags::Mortars<SomeTag, 3>>(
      "Mortars(SomeTag)");
  TestHelpers::db::test_simple_tag<Tags::MortarSize<2>>("MortarSize");
  TestHelpers::db::test_simple_tag<Tags::NumericalFlux<SomeType>>(
      "NumericalFlux");
}
