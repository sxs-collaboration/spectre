// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "Elliptic/Tags.hpp"

SPECTRE_TEST_CASE("Unit.Elliptic.Tags", "[Unit][Elliptic]") {
  CHECK(Elliptic::Tags::IterationId<>::name() == "EllipticIterationId");
}
