// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "Elliptic/Systems/Xcts/Tags.hpp"

struct DataVector;
namespace Frame {
struct Inertial;
}  // namespace Frame

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.Xcts.Tags", "[Unit][Elliptic]") {
  CHECK(Xcts::Tags::ConformalFactor<DataVector>::name() == "ConformalFactor");
  CHECK(Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial,
                                            DataVector>::name() ==
        "ConformalFactorGradient");
}
