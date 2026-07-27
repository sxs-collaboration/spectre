// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/RadiationTransport/NoNeutrinos/System.hpp"
#include "Time/ChangeTimeStepperOrder.tpp"
#include "Time/CleanHistory.tpp"
#include "Time/RecordTimeStepperData.tpp"
#include "Time/UpdateU.tpp"
#include "Utilities/GenerateInstantiations.hpp"

#define NEUTRINO(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                            \
  template class ChangeTimeStepperOrder<                  \
      grmhd::GhValenciaDivClean::System<NEUTRINO(data)>>; \
  template class CleanHistory<                            \
      grmhd::GhValenciaDivClean::System<NEUTRINO(data)>>; \
  template class RecordTimeStepperData<                   \
      grmhd::GhValenciaDivClean::System<NEUTRINO(data)>>; \
  template class UpdateU<grmhd::GhValenciaDivClean::System<NEUTRINO(data)>>;

GENERATE_INSTANTIATIONS(INSTANTIATION,
                        (RadiationTransport::NoNeutrinos::System))

#undef INSTANTIATION
#undef NEUTRINO
