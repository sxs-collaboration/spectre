// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "Domain/BoundaryVariablesTag.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/TestTags.hpp"
#include "Utilities/TMPL.hpp"

namespace {
SPECTRE_TEST_CASE("Unit.Domain.BoundaryVariablesTag", "[Unit][Domain]") {
  TestHelpers::db::test_simple_tag<Tags::BoundaryVariables<
      2, tmpl::list<TestHelpers::Tags::Vector<DataVector>,
                    TestHelpers::Tags::Scalar<DataVector>,
                    TestHelpers::Tags::Scalar2<DataVector>>>>(
      "BoundaryVariables(Vector,Scalar,Scalar2)");
}
}  // namespace
