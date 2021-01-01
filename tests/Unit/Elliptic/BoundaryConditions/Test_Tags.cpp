// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Elliptic/BoundaryConditions/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
struct FieldTag {};
}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.BoundaryConditions.Tags", "[Unit][Elliptic]") {
  TestHelpers::db::test_prefix_tag<
      elliptic::Tags::BoundaryConditionType<FieldTag>>(
      "BoundaryConditionType(FieldTag)");
  TestHelpers::db::test_simple_tag<
      elliptic::Tags::BoundaryConditionTypes<tmpl::list<FieldTag>>>(
      "BoundaryConditionTypes");
}
