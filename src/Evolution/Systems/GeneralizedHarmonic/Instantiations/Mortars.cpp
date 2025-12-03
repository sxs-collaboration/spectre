// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/CleanMortarHistory.tpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Utilities/GenerateInstantiations.hpp"

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class evolution::dg::CleanMortarHistory<gh::System<DIM(data)>>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
