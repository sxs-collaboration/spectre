// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Framework/TestCreation.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/AngularOrdering.hpp"
#include "Utilities/GetOutput.hpp"

namespace ylm {
SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.SphericalHarmonics.AngularOrdering",
                  "[Unit]") {
  CHECK(get_output(AngularOrdering::Cce) == "Cce");
  CHECK(get_output(AngularOrdering::Strahlkorper) == "Strahlkorper");
  CHECK(TestHelpers::test_creation<AngularOrdering>("Cce") ==
        AngularOrdering::Cce);
  CHECK(TestHelpers::test_creation<AngularOrdering>("Strahlkorper") ==
        AngularOrdering::Strahlkorper);
}
}  // namespace ylm
