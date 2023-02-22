// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace domain {
namespace {
template <size_t Dim>
void test_simple_tags() {
  TestHelpers::db::test_simple_tag<Tags::Domain<Dim>>("Domain");
  TestHelpers::db::test_simple_tag<Tags::InitialExtents<Dim>>("InitialExtents");
}

SPECTRE_TEST_CASE("Unit.Domain.Creators.Tags", "[Unit][Domain]") {
  test_simple_tags<1>();
  test_simple_tags<2>();
  test_simple_tags<3>();
}
}  // namespace
}  // namespace domain
