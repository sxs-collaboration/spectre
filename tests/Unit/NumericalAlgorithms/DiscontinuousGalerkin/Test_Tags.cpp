// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"

namespace {
struct SomeTag : db::SimpleTag {
  using type = int;
  static std::string name() noexcept { return "SomeTag"; }
};
}  // namespace

SPECTRE_TEST_CASE("Unit.DG.Tags", "[Unit][NumericalAlgorithms]") {
  CHECK(db::tag_name<::Tags::Mortars<SomeTag, 3>>() == "Mortars(SomeTag)");
}
