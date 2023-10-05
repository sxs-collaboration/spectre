// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Executables/GeneralizedHarmonic/EvolveGhAndCharacteristic.hpp"

#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/RegisterDerivedWithCharm.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/CubicSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/LinearSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Parallel/CharmMain.tpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

using metavariables = EvolutionMetavars<EVOLVE_CCM>;

extern "C" void CkRegisterMainModule() {
  Parallel::charmxx::register_main_module<metavariables>();
  Parallel::charmxx::register_init_node_and_proc(
      {&domain::creators::register_derived_with_charm,
       &domain::creators::time_dependence::register_derived_with_charm,
       &domain::FunctionsOfTime::register_derived_with_charm,
       &gh::BoundaryCorrections::register_derived_with_charm,
       &gh::ConstraintDamping::register_derived_with_charm,
       &Cce::register_initialize_j_with_charm<
           metavariables::evolve_ccm, metavariables::cce_boundary_component>,
       &register_derived_classes_with_charm<Cce::WorldtubeDataManager>,
       &register_derived_classes_with_charm<intrp::SpanInterpolator>,
       &register_factory_classes_with_charm<metavariables>},
      {});
}
