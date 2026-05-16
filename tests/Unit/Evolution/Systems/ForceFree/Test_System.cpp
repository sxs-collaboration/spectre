// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/ForceFree/System.hpp"

SPECTRE_TEST_CASE("Unit.ForceFree.System.Name", "[Unit][ForceFree]") {
  CHECK(ForceFree::System::name() == "ForceFree");
}
