// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Executables/CurvedScalarWave/EvolveWorldtubeCurvedScalarWave.hpp"

#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryCorrections/RegisterDerived.hpp"
#include "Parallel/CharmMain.tpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

using metavariables = EvolutionMetavars<gr::Solutions::KerrSchild,
                                        ScalarWave::Solutions::PlaneWave<3>>;

extern "C" void CkRegisterMainModule() {
  Parallel::charmxx::register_main_module<metavariables>();
  Parallel::charmxx::register_init_node_and_proc(
      {&domain::creators::register_derived_with_charm,
       &domain::creators::time_dependence::register_derived_with_charm,
       &domain::FunctionsOfTime::register_derived_with_charm,
       &CurvedScalarWave::BoundaryCorrections::register_derived_with_charm,
       &register_factory_classes_with_charm<metavariables>},
      {});
}
