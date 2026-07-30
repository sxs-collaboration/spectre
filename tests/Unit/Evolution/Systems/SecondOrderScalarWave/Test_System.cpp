// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/SecondOrderScalarWave/System.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.SecondOrderScalarWave.System",
                  "[Unit][Evolution]") {
  CHECK(SecondOrderScalarWave::System<1>::name() == "SecondOrderScalarWave");
  CHECK(SecondOrderScalarWave::System<2>::name() == "SecondOrderScalarWave");
  CHECK(SecondOrderScalarWave::System<3>::name() == "SecondOrderScalarWave");
}
