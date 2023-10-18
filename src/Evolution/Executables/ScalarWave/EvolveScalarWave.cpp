// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Executables/ScalarWave/EvolveScalarWave.hpp"

#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/ScalarWave/BoundaryCorrections/RegisterDerived.hpp"
#include "Parallel/CharmMain.tpp"
#include "ParallelAlgorithms/Amr/Actions/RegisterCallbacks.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

// Chosen in CMakeLists.txt
using metavariables = EvolutionMetavars<DIM>;

extern "C" void CkRegisterMainModule() {
  Parallel::charmxx::register_main_module<metavariables>();
  Parallel::charmxx::register_init_node_and_proc(
      {&domain::creators::register_derived_with_charm,
       &domain::creators::time_dependence::register_derived_with_charm,
       &domain::FunctionsOfTime::register_derived_with_charm,
       &ScalarWave::BoundaryCorrections::register_derived_with_charm,
       &register_factory_classes_with_charm<metavariables>,
       &amr::register_callbacks<metavariables,
                                typename metavariables::dg_element_array>},
      {});
}
