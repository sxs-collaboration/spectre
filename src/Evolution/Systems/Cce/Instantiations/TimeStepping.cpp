// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/KleinGordonSystem.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/System.hpp"
#include "Time/ChangeTimeStepperOrder.tpp"
#include "Time/CleanHistory.tpp"
#include "Time/RecordTimeStepperData.tpp"
#include "Time/UpdateU.tpp"
#include "Utilities/GenerateInstantiations.hpp"

#define EVOLVE_CCM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                          \
  template class ChangeTimeStepperOrder<                                \
      Cce::KleinGordonSystem<EVOLVE_CCM(data)>,                         \
      Cce::Tags::CceEvolutionPrefix>;                                   \
  template class CleanHistory<Cce::KleinGordonSystem<EVOLVE_CCM(data)>, \
                              Cce::Tags::CceEvolutionPrefix>;           \
  template class RecordTimeStepperData<                                 \
      Cce::KleinGordonSystem<EVOLVE_CCM(data)>>;                        \
  template class UpdateU<Cce::KleinGordonSystem<EVOLVE_CCM(data)>,      \
                         Cce::Tags::CceEvolutionPrefix>;                \
  template class ChangeTimeStepperOrder<Cce::System<EVOLVE_CCM(data)>,  \
                                        Cce::Tags::CceEvolutionPrefix>; \
  template class CleanHistory<Cce::System<EVOLVE_CCM(data)>,            \
                              Cce::Tags::CceEvolutionPrefix>;           \
  template class RecordTimeStepperData<Cce::System<EVOLVE_CCM(data)>>;  \
  template class UpdateU<Cce::System<EVOLVE_CCM(data)>,                 \
                         Cce::Tags::CceEvolutionPrefix>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (false, true))

#undef INSTANTIATION
#undef DIM
