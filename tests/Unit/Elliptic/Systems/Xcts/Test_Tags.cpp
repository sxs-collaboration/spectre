// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

struct DataVector;
namespace Frame {
struct Inertial;
}  // namespace Frame

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.Xcts.Tags", "[Unit][Elliptic]") {
  TestHelpers::db::test_simple_tag<Xcts::Tags::ConformalFactor<DataVector>>(
      "ConformalFactor");
  TestHelpers::db::test_simple_tag<
      Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial, DataVector>>(
      "ConformalFactorGradient");
}
