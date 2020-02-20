// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "Elliptic/Tags.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"

namespace {
struct SomeFluxesComputer {};
}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Tags", "[Unit][Elliptic]") {
  TestHelpers::db::test_simple_tag<
      elliptic::Tags::FluxesComputer<SomeFluxesComputer>>("SomeFluxesComputer");
}
