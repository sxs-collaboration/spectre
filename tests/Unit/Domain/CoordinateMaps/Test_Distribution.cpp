// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Framework/TestCreation.hpp"
#include "Utilities/GetOutput.hpp"

namespace domain::CoordinateMaps {

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Distribution", "[Domain][Unit]") {
  CHECK(get_output(Distribution::Linear) == "Linear");
  CHECK(get_output(Distribution::Equiangular) == "Equiangular");
  CHECK(get_output(Distribution::Logarithmic) == "Logarithmic");
  CHECK(get_output(Distribution::Inverse) == "Inverse");
  CHECK(get_output(Distribution::Projective) == "Projective");
  CHECK(TestHelpers::test_creation<Distribution>("Linear") ==
        Distribution::Linear);
  CHECK(TestHelpers::test_creation<Distribution>("Equiangular") ==
        Distribution::Equiangular);
  CHECK(TestHelpers::test_creation<Distribution>("Logarithmic") ==
        Distribution::Logarithmic);
  CHECK(TestHelpers::test_creation<Distribution>("Inverse") ==
        Distribution::Inverse);
  CHECK(TestHelpers::test_creation<Distribution>("Projective") ==
        Distribution::Projective);
}

}  // namespace domain::CoordinateMaps
