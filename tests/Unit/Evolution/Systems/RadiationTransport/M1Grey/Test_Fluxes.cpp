// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Fluxes.hpp"
#include "Evolution/Systems/RadiationTransport/Tags.hpp"  // IWYU pragma: keep
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

SPECTRE_TEST_CASE("Unit.RadiationTransport.M1Grey.Fluxes", "[Unit][M1Grey]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/RadiationTransport/M1Grey"};

  pypp::check_with_random_values<1>(&RadiationTransport::M1Grey::ComputeFluxes<
                                        neutrinos::ElectronNeutrinos<0>>::apply,
                                    "Fluxes", {"tilde_e_flux", "tilde_s_flux"},
                                    {{{0.0, 1.0}}}, DataVector{5});
}
