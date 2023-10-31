// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Executables/Xcts/SolveXcts.hpp"

#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Elliptic/SubdomainPreconditioners/RegisterDerived.hpp"
#include "Parallel/CharmMain.tpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/RegisterDerivedWithCharm.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

extern "C" void CkRegisterMainModule() {
  Parallel::charmxx::register_main_module<Metavariables>();
  Parallel::charmxx::register_init_node_and_proc(
      {&domain::creators::register_derived_with_charm,
       &domain::FunctionsOfTime::register_derived_with_charm,
       &register_derived_classes_with_charm<
           Metavariables::solver::schwarz_smoother::subdomain_solver>,
       &elliptic::subdomain_preconditioners::register_derived_with_charm,
       &EquationsOfState::register_derived_with_charm,
       &register_factory_classes_with_charm<Metavariables>},
      {});
}
