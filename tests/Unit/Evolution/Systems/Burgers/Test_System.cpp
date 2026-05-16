// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/Burgers/System.hpp"

SPECTRE_TEST_CASE("Unit.Burgers.System.Name", "[Unit][Burgers]") {
  CHECK(Burgers::System::name() == "Burgers");
}
