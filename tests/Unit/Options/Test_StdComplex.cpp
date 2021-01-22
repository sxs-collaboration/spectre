// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <complex>

#include "Framework/TestCreation.hpp"
#include "Options/StdComplex.hpp"

SPECTRE_TEST_CASE("Unit.Options.StdParsers", "[Unit][Options]") {
  CHECK(std::complex<double>(1.2, 2.4) ==
        TestHelpers::test_creation<std::complex<double>>("[1.2, 2.4]"));
}
