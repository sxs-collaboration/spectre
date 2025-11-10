// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/InboxBoundaryData.hpp"

#include <cstddef>
#include <pup.h>
#include <pup_stl.h>

#include "Utilities/GenerateInstantiations.hpp"

namespace evolution::dg {
template <size_t Dim>
bool InboxBoundaryData<Dim>::empty() const {
  return messages.empty();
}

template <size_t Dim>
void InboxBoundaryData<Dim>::collect_messages() {}

template <size_t Dim>
void InboxBoundaryData<Dim>::pup(PUP::er& p) {
  p | messages;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template struct InboxBoundaryData<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg
