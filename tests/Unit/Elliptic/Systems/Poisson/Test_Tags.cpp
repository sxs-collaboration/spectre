// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "Elliptic/Systems/Poisson/Tags.hpp"

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.Poisson.Tags", "[Unit][Elliptic]") {
  CHECK(db::tag_name<Poisson::Tags::Field>() == "Field");
}
