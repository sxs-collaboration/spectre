// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/ForceFree/Fluxes.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ForceFree.Fluxes",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ForceFree"};

  pypp::check_with_random_values<1>(
      &ForceFree::Fluxes::apply, "Fluxes",
      {"tilde_e_flux", "tilde_b_flux", "tilde_psi_flux", "tilde_phi_flux",
       "tilde_q_flux"},
      {{{0.0, 1.0}}}, DataVector{5});
}
