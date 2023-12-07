// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Executables/GrMhd/GhValenciaDivClean/EvolveGhValenciaDivClean.hpp"

#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/RegisterDerivedWithCharm.hpp"
#include "Parallel/CharmMain.tpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/RegisterDerivedWithCharm.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

// Parameters chosen in CMakeLists.txt
using metavariables = EvolutionMetavars<USE_CONTROL_SYSTEMS, BondiSachs>;

extern "C" void CkRegisterMainModule() {
  Parallel::charmxx::register_main_module<metavariables>();
  Parallel::charmxx::register_init_node_and_proc(
      {&domain::creators::register_derived_with_charm,
       &domain::creators::time_dependence::register_derived_with_charm,
       &domain::FunctionsOfTime::register_derived_with_charm,
       &grmhd::GhValenciaDivClean::BoundaryCorrections::
           register_derived_with_charm,
       &grmhd::GhValenciaDivClean::fd::register_derived_with_charm,
       &EquationsOfState::register_derived_with_charm,
       &gh::ConstraintDamping::register_derived_with_charm,
       &register_factory_classes_with_charm<metavariables>},
      {});
}
