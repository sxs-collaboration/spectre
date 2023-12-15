// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Imex/Mode.hpp"
#include "Framework/TestCreation.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Imex.Mode", "[Unit][Evolution]") {
  CHECK(TestHelpers::test_creation<imex::Mode>("Implicit") ==
        imex::Mode::Implicit);
  CHECK(TestHelpers::test_creation<imex::Mode>("SemiImplicit") ==
        imex::Mode::SemiImplicit);
}
