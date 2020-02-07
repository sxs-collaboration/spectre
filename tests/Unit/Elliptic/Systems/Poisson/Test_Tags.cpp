// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.Poisson.Tags", "[Unit][Elliptic]") {
  TestHelpers::db::test_simple_tag<Poisson::Tags::Field> ("Field");
}
