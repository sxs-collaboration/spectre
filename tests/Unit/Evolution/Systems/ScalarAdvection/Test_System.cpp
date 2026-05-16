// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/ScalarAdvection/System.hpp"

SPECTRE_TEST_CASE("Unit.ScalarAdvection.System.Name",
                  "[Unit][ScalarAdvection]") {
  CHECK(ScalarAdvection::System<1>::name() == "ScalarAdvection");
  CHECK(ScalarAdvection::System<2>::name() == "ScalarAdvection");
}
