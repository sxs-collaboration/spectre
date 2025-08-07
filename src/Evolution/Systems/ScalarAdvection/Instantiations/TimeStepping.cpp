// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarAdvection/System.hpp"
#include "Time/RecordTimeStepperData.tpp"
#include "Time/UpdateU.tpp"
#include "Utilities/GenerateInstantiations.hpp"

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                              \
  template class RecordTimeStepperData<ScalarAdvection::System<DIM(data)>>; \
  template class UpdateU<ScalarAdvection::System<DIM(data)>, false>;        \
  template class UpdateU<ScalarAdvection::System<DIM(data)>, true>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
