// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/CurvedScalarWave/System.hpp"

SPECTRE_TEST_CASE("Unit.CurvedScalarWave.System.Name", "[Unit][Evolution]") {
  CHECK(CurvedScalarWave::System<1>::name() == "CurvedScalarWave");
  CHECK(CurvedScalarWave::System<2>::name() == "CurvedScalarWave");
  CHECK(CurvedScalarWave::System<3>::name() == "CurvedScalarWave");
}
