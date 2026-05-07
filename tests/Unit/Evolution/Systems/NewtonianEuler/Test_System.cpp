// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/NewtonianEuler/System.hpp"

SPECTRE_TEST_CASE("Unit.NewtonianEuler.System.Name", "[Unit][Evolution]") {
  CHECK(NewtonianEuler::System<1>::name() == "NewtonianEuler");
  CHECK(NewtonianEuler::System<2>::name() == "NewtonianEuler");
  CHECK(NewtonianEuler::System<3>::name() == "NewtonianEuler");
}
