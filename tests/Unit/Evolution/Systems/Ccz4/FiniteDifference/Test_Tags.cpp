// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
struct ArbitraryFrame;

void test_simple_tags() {
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::Reconstructor>(
      "Reconstructor");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::EvolveLapseAndShift>(
      "EvolveLapseAndShift");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::ConstrainedEvolution>(
      "ConstrainedEvolution");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::KreissOligerEpsilon>(
      "KreissOligerEpsilon");
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.fd.Ccz4.Tags", "[Unit][Evolution]") {
  test_simple_tags();
}
}  // namespace
