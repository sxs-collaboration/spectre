// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Executables/Cce/CharacteristicExtract.hpp"

#include <vector>

#include "Evolution/Systems/Cce/Initialize/RegisterInitializeJWithCharm.hpp"
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
       &register_derived_classes_with_charm<Cce::WorldtubeBufferUpdater<
           Cce::Tags::worldtube_boundary_tags_for_writing<
               Spectral::Swsh::Tags::SwshTransform>>>,
       &register_derived_classes_with_charm<Cce::WorldtubeDataManager<
           Cce::Tags::characteristic_worldtube_boundary_tags<
               Cce::Tags::BoundaryValue>>>,
       &register_derived_classes_with_charm<intrp::SpanInterpolator>,
       &register_derived_classes_with_charm<Cce::Solutions::WorldtubeData>,
       &register_factory_classes_with_charm<metavariables>},
      {});
}
