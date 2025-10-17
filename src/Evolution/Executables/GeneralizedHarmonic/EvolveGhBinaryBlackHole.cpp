// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Executables/GeneralizedHarmonic/EvolveGhBinaryBlackHole.hpp"

#include <vector>

#include "ControlSystem/ControlErrors/Size/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Parallel/CharmMain.tpp"
#include "ParallelAlgorithms/Amr/Actions/RegisterCallbacks.hpp"
#include "PointwiseFunctions/ConstraintDamping/RegisterDerivedWithCharm.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/System/AttachDebugger.hpp"

extern "C" void CkRegisterMainModule() {
  Parallel::charmxx::register_main_module<EvolutionMetavars>();
  Parallel::charmxx::register_init_node_and_proc(
      {&sys::attach_debugger, &domain::creators::register_derived_with_charm,
       &domain::creators::time_dependence::register_derived_with_charm,
       &domain::FunctionsOfTime::register_derived_with_charm,
       &ConstraintDamping::register_derived_with_charm,
       &control_system::size::register_derived_with_charm,
       &register_factory_classes_with_charm<EvolutionMetavars>,
       &amr::register_callbacks<EvolutionMetavars,
                                EvolutionMetavars::gh_dg_element_array>},
      {});
}
