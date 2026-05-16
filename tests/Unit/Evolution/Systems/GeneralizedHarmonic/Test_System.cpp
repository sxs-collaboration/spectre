// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"

SPECTRE_TEST_CASE("Unit.gh.System.Name", "[Unit][Evolution]") {
  CHECK(gh::System<1>::name() == "GeneralizedHarmonic");
  CHECK(gh::System<2>::name() == "GeneralizedHarmonic");
  CHECK(gh::System<3>::name() == "GeneralizedHarmonic");
}
