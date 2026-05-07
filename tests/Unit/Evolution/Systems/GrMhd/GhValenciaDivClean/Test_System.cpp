// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/System.hpp"
#include "Evolution/Systems/RadiationTransport/NoNeutrinos/System.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE("Unit.grmhd.GhValenciaDivClean.System.Name",
                  "[Unit][Evolution]") {
  using M1Transport = RadiationTransport::M1Grey::System<
      tmpl::list<neutrinos::ElectronNeutrinos<1>>>;
  using NoNuTransport = RadiationTransport::NoNeutrinos::System;
  CHECK(grmhd::GhValenciaDivClean::System<M1Transport>::name() ==
        "GhValenciaDivClean");
  CHECK(grmhd::GhValenciaDivClean::System<NoNuTransport>::name() ==
        "GhValenciaDivClean");
}
