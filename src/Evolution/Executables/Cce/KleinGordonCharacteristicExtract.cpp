// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Executables/Cce/KleinGordonCharacteristicExtract.hpp"

#include <vector>

#include "Evolution/Systems/Cce/Initialize/RegisterInitializeJWithCharm.hpp"
#include "Evolution/Systems/Cce/WorldtubeBufferUpdater.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/CubicSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/LinearSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Parallel/CharmMain.tpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

// Parameters chosen in CMakeLists.txt
using metavariables = EvolutionMetavars<BOUNDARY_COMPONENT>;

extern "C" void CkRegisterMainModule() {
  Parallel::charmxx::register_main_module<metavariables>();
  Parallel::charmxx::register_init_node_and_proc(
      {&Cce::register_initialize_j_with_charm<
           metavariables::evolve_ccm, metavariables::cce_boundary_component>,
       &register_derived_classes_with_charm<
           Cce::WorldtubeBufferUpdater<Cce::cce_metric_input_tags>>,
       &register_derived_classes_with_charm<
           Cce::WorldtubeBufferUpdater<Cce::cce_bondi_input_tags>>,
       &register_derived_classes_with_charm<
           Cce::WorldtubeBufferUpdater<Cce::klein_gordon_input_tags>>,
       &register_derived_classes_with_charm<Cce::WorldtubeDataManager<
           Cce::Tags::characteristic_worldtube_boundary_tags<
               Cce::Tags::BoundaryValue>>>,
       &register_derived_classes_with_charm<Cce::WorldtubeDataManager<
           Cce::Tags::klein_gordon_worldtube_boundary_tags>>,
       &register_derived_classes_with_charm<intrp::SpanInterpolator>,
       &register_derived_classes_with_charm<Cce::Solutions::WorldtubeData>,
       &register_derived_classes_with_charm<
           Cce::Solutions::KleinGordonWorldtubeData>,
       &register_factory_classes_with_charm<metavariables>},
      {});
}
