// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Amr/Tags/NeighborFlags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
template <size_t Dim>
void test() {
  TestHelpers::db::test_simple_tag<amr::Tags::Flags<Dim>>("Flags");
  TestHelpers::db::test_simple_tag<amr::Tags::NeighborFlags<Dim>>(
      "NeighborFlags");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Amr.Tags", "[Domain][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
