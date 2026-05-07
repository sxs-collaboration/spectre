// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/ScalarWave/System.hpp"

SPECTRE_TEST_CASE("Unit.ScalarWave.System.Name", "[Unit][Evolution]") {
  CHECK(ScalarWave::System<1>::name() == "ScalarWave");
  CHECK(ScalarWave::System<2>::name() == "ScalarWave");
  CHECK(ScalarWave::System<3>::name() == "ScalarWave");
}
