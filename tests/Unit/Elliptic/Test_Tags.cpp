// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "Elliptic/Tags.hpp"

namespace {
struct SomeFluxesComputer {};
}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Tags", "[Unit][Elliptic]") {
  CHECK(db::tag_name<elliptic::Tags::FluxesComputer<SomeFluxesComputer>>() ==
        "SomeFluxesComputer");
}
