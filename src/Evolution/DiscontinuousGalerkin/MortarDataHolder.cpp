// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/MortarDataHolder.hpp"

#include <pup.h>

#include "Utilities/GenerateInstantiations.hpp"

namespace evolution::dg {
template <size_t Dim>
void MortarDataHolder<Dim>::pup(PUP::er& p) {
  p | local_data_;
  p | neighbor_data_;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template class MortarDataHolder<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg
