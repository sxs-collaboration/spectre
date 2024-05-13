// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/AtomicInboxBoundaryData.hpp"

#include <cstddef>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace evolution::dg {
template <size_t Dim>
size_t AtomicInboxBoundaryData<Dim>::index(
    const DirectionalId<Dim>& neighbor_directional_id) {
  if constexpr (Dim == 1) {
    // Note that in 1d:
    //     pow<Dim>(2) * neighbor_directional_id.direction.dimension() == 0
    //     pow<Dim - 1>(2) == 1
    // so:
    //   pow<Dim>(2) * neighbor_directional_id.direction.dimension() +
    //   pow<Dim - 1>(2) *
    //     (neighbor_directional_id.direction.side() == Side::Lower ? 0 : 1)
    //
    // is just:
    //    (neighbor_directional_id.direction.side() == Side::Lower ? 0 : 1)
    return neighbor_directional_id.direction().side() == Side::Lower ? 0_st
                                                                     : 1_st;
  } else {
    size_t result = 0;
    for (size_t i = 0, j = 0; i < Dim; ++i) {
      if (i == neighbor_directional_id.direction().dimension()) {
        continue;
      }
      result = result | (neighbor_directional_id.id().segment_id(i).index() & 1)
                            << j;
      ++j;
    }
    return pow<Dim>(2_st) * neighbor_directional_id.direction().dimension() +
           pow<Dim - 1>(2_st) *
               (neighbor_directional_id.direction().side() == Side::Lower
                    ? 0_st
                    : 1_st) +
           result;
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class AtomicInboxBoundaryData<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg
