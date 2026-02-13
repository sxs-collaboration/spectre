// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Time/BoundaryHistory.tpp"
#include "Utilities/GenerateInstantiations.hpp"

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                           \
  INSTANTIATE_BOUNDARY_HISTORY(::evolution::dg::MortarData<DIM(data)>, \
                               ::evolution::dg::MortarData<DIM(data)>, \
                               DataVector)

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
