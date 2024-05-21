// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/AtomicInboxBoundaryData.hpp"

#include <atomic>
#include <cstddef>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/System/Abort.hpp"

namespace evolution::dg {
template <size_t Dim>
AtomicInboxBoundaryData<Dim>::AtomicInboxBoundaryData(
    AtomicInboxBoundaryData<Dim>&& rhs) noexcept {
  if (rhs.message_count.load(std::memory_order_acquire) != 0) {
    sys::abort(
        "You cannot move an AtomicInboxBoundaryData with non-zero message "
        "count.");
  }
  for (size_t i = 0; i < rhs.boundary_data_in_directions.size(); ++i) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    if (not rhs.boundary_data_in_directions[i].empty()) {
      sys::abort(
          "You cannot move an AtomicInboxBoundaryData with data in "
          "boundary_data_in_directions.");
    }
  }
  message_count.store(0, std::memory_order_release);
  number_of_neighbors.store(
      rhs.number_of_neighbors.load(std::memory_order_acquire),
      std::memory_order_release);
}

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

template <size_t Dim>
void AtomicInboxBoundaryData<Dim>::pup(PUP::er& p) {
  if (UNLIKELY(number_of_neighbors.load(std::memory_order_acquire) != 0)) {
    ERROR(
        "Can only serialize AtomicInboxBoundaryData if there are no messages. "
        "We need to be very careful about serializing atomics since "
        "serialization requires strong synchronization like a lock.");
  }
  for (size_t i = 0; i < boundary_data_in_directions.size(); ++i) {
    if (UNLIKELY(not gsl::at(boundary_data_in_directions, i).empty())) {
      ERROR(
          "We can only serialize empty StaticSpscQueues but the queue in "
          "element "
          << i << " is not empty.");
    }
  }
  if (p.isUnpacking()) {
    std::atomic_uint::value_type number_of_neighbors_to_serialize = 0;
    p | number_of_neighbors_to_serialize;
    number_of_neighbors.store(number_of_neighbors_to_serialize,
                              std::memory_order_release);
    message_count.store(0, std::memory_order_release);
  } else {
    std::atomic_uint::value_type number_of_neighbors_to_serialize =
        number_of_neighbors.load(std::memory_order_acquire);
    p | number_of_neighbors_to_serialize;
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class AtomicInboxBoundaryData<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg
