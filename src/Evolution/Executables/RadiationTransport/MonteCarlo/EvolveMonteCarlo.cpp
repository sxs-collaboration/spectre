// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Executables/RadiationTransport/MonteCarlo/EvolveMonteCarlo.hpp"

#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Parallel/CharmMain.tpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/RegisterDerivedWithCharm.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

void register_neutrino_tables() {
  register_classes_with_charm(
      tmpl::list<Particles::MonteCarlo::NeutrinoInteractionTable<2, 2>,
                 Particles::MonteCarlo::NeutrinoInteractionTable<2, 3>,
                 Particles::MonteCarlo::NeutrinoInteractionTable<4, 3>,
                 Particles::MonteCarlo::NeutrinoInteractionTable<16, 3>>{});
}

void register_mc_options() {
  register_classes_with_charm(
      tmpl::list<Particles::MonteCarlo::MonteCarloOptions<2>,
                 Particles::MonteCarlo::MonteCarloOptions<3>>{});
}

extern "C" void CkRegisterMainModule() {
  Parallel::charmxx::register_main_module<EvolutionMetavars<4, 3>>();
  Parallel::charmxx::register_init_node_and_proc(
      {&domain::creators::register_derived_with_charm,
       &domain::creators::time_dependence::register_derived_with_charm,
       &domain::FunctionsOfTime::register_derived_with_charm,
       &EquationsOfState::register_derived_with_charm,
       &register_factory_classes_with_charm<EvolutionMetavars<4, 3>>,
       &register_neutrino_tables, &register_mc_options},
      {});
}
