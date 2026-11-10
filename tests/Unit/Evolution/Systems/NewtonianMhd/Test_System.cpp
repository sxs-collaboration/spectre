// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/NewtonianMhd/System.hpp"

SPECTRE_TEST_CASE("Unit.NewtonianMhd.System.Name", "[Unit][Evolution]") {
  CHECK(NewtonianMhd::System<1>::name() == "NewtonianEuler");
  CHECK(NewtonianMhd::System<2>::name() == "NewtonianEuler");
  CHECK(NewtonianMhd::System<3>::name() == "NewtonianEuler");
}
