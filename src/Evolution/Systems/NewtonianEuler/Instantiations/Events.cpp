// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/System.hpp"
#include "ParallelAlgorithms/Events/ObserveTimeStep.tpp"
#include "Utilities/GenerateInstantiations.hpp"

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class Events::ObserveTimeStep<NewtonianEuler::System<DIM(data)>>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
