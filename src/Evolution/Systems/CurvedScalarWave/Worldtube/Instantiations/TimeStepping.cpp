// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Worldtube/System.hpp"
#include "Time/ChangeTimeStepperOrder.tpp"
#include "Time/CleanHistory.tpp"
#include "Time/RecordTimeStepperData.tpp"
#include "Time/UpdateU.tpp"
#include "Utilities/GenerateInstantiations.hpp"

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                 \
  template class ChangeTimeStepperOrder<                                       \
      CurvedScalarWave::Worldtube::System<DIM(data)>>;                         \
  template class CleanHistory<CurvedScalarWave::Worldtube::System<DIM(data)>>; \
  template class RecordTimeStepperData<                                        \
      CurvedScalarWave::Worldtube::System<DIM(data)>>;                         \
  template class UpdateU<CurvedScalarWave::Worldtube::System<DIM(data)>,       \
                         false>;                                               \
  template class UpdateU<CurvedScalarWave::Worldtube::System<DIM(data)>, true>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (3))

#undef INSTANTIATION
#undef DIM
