// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/M1HydroCoupling.hpp"
#include "Evolution/Systems/RadiationTransport/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Utilities/Gsl.hpp"

SPECTRE_TEST_CASE("Unit.RadiationTransport.M1Grey.M1HydroCoupling",
                  "[Unit][M1Grey]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/RadiationTransport/M1Grey"};

  using neutrino_species = tmpl::list<neutrinos::ElectronNeutrinos<1>>;
  using CouplingClass =
      typename RadiationTransport::M1Grey::ComputeM1HydroCoupling<
          neutrino_species>;

  using CouplingJacobianClass =
      typename RadiationTransport::M1Grey::ComputeM1HydroCouplingJacobian<
          neutrino_species>;

  pypp::check_with_random_values<1>(
      &CouplingClass::apply, "M1HydroCoupling",
      {"hydro_coupling_tilde_e", "hydro_coupling_tilde_s"}, {{{0.0, 1.0}}},
      DataVector{5});

  pypp::check_with_random_values<1>(
      &CouplingJacobianClass::apply, "ComputeM1HydroCouplingJacobian",
      {"hydro_coupling_jacobian_deriv_e_source_e",
       "hydro_coupling_jacobian_deriv_e_source_s",
       "hydro_coupling_jacobian_deriv_s_source_e",
       "hydro_coupling_jacobian_deriv_s_source_s"},
      {{{0.0, 1.0}}}, DataVector{5});
}
