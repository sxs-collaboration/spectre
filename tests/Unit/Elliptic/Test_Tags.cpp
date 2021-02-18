// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Elliptic/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
struct SomeFluxesComputer {};
struct Dummy {};
}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Tags", "[Unit][Elliptic]") {
  TestHelpers::db::test_simple_tag<
      elliptic::Tags::FluxesComputer<SomeFluxesComputer>>("SomeFluxesComputer");
  TestHelpers::db::test_simple_tag<elliptic::Tags::Background<Dummy>>(
      "Background");
  TestHelpers::db::test_simple_tag<elliptic::Tags::InitialGuess<Dummy>>(
      "InitialGuess");
}
