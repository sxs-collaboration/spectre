// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Elliptic/DiscontinuousGalerkin/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace elliptic::dg {

SPECTRE_TEST_CASE("Unit.Elliptic.DG.Tags", "[Unit][Elliptic]") {
  TestHelpers::db::test_simple_tag<Tags::PenaltyParameter>("PenaltyParameter");
}

}  // namespace elliptic::dg
