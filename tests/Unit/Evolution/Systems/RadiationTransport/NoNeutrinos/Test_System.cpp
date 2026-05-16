// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/RadiationTransport/NoNeutrinos/System.hpp"

SPECTRE_TEST_CASE("Unit.RadiationTransport.NoNeutrinos.System.Name",
                  "[Unit][Evolution]") {
  CHECK(RadiationTransport::NoNeutrinos::System::name() == "NoNeutrinos");
}
