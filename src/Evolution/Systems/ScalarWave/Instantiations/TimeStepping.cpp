// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/System.hpp"
#include "Time/CleanHistory.tpp"
#include "Time/RecordTimeStepperData.tpp"
#include "Time/UpdateU.tpp"
#include "Utilities/GenerateInstantiations.hpp"

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                         \
  template class CleanHistory<ScalarWave::System<DIM(data)>>;          \
  template class RecordTimeStepperData<ScalarWave::System<DIM(data)>>; \
  template class UpdateU<ScalarWave::System<DIM(data)>, false>;        \
  template class UpdateU<ScalarWave::System<DIM(data)>, true>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
