// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/AtomicInboxBoundaryData.hpp"

#include <atomic>
#include <cstddef>
#include <pup.h>
#include <pup_stl.h>
#include <utility>

#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/System/Abort.hpp"

namespace evolution::dg {
template <size_t Dim>
AtomicInboxBoundaryData<Dim>::AtomicInboxBoundaryData(
    AtomicInboxBoundaryData<Dim>&& rhs) noexcept {
  if (rhs.missing_messages.load(std::memory_order_acquire) !=
          rhs.passed_missing_messages - rhs.processed_messages or
      not messages.empty()) {
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
  missing_messages.store(0, std::memory_order_release);
  processed_messages = 0;
  passed_missing_messages = 0;
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
void AtomicInboxBoundaryData<Dim>::collect_messages() {
  for (auto& spsc_in_direction : boundary_data_in_directions) {
    auto* data_in_direction = spsc_in_direction.front();
    while (data_in_direction != nullptr) {
      const auto& time_step_id = get<0>(*data_in_direction);
      auto& data = get<1>(*data_in_direction);
      auto& directional_element_id = get<2>(*data_in_direction);
      auto& current_inbox = messages[time_step_id];
      if (auto it = current_inbox.find(directional_element_id);
          it != current_inbox.end()) {
        merge_boundary_data(make_not_null(&it->second), std::move(data));
      } else {
        // We have not received ghost cells or fluxes at this time.
        if (not current_inbox
                    .emplace(std::move(directional_element_id), std::move(data))
                    .second) {
          ERROR("Failed to insert data to receive at instance '"
                << time_step_id
                << "' with tag 'BoundaryCorrectionAndGhostCellsInbox'.\n");
        }
      }

      spsc_in_direction.pop();
      data_in_direction = spsc_in_direction.front();
      ++processed_messages;
    }  // while data_in_direction != nullptr
  }  //   for spsc_in_direction : boundary_data_in_directions
}

template <size_t Dim>
bool AtomicInboxBoundaryData<Dim>::set_missing_messages(const size_t count) {
  const int old_missing_messages = missing_messages.fetch_add(
      static_cast<int>(count) + processed_messages - passed_missing_messages,
      std::memory_order_acq_rel);
  const int queued_messages =
      passed_missing_messages - old_missing_messages - processed_messages;
  processed_messages = 0;
  passed_missing_messages = static_cast<int>(count);
  return queued_messages >= static_cast<int>(count);
}

template <size_t Dim>
void AtomicInboxBoundaryData<Dim>::pup(PUP::er& p) {
  const auto missing = missing_messages.load(std::memory_order_acquire);
  if (UNLIKELY(missing > 0 or
               missing != passed_missing_messages - processed_messages or
               not messages.empty())) {
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
    // We only need to preserve the combination of these quantities
    // representing the number of queued messages, which we checked
    // during serialization was zero, and the fact that we are not
    // waiting on messages.  Exactly how it was split between the
    // fields is unimportant.
    missing_messages.store(0, std::memory_order_release);
    processed_messages = 0;
    passed_missing_messages = 0;
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class AtomicInboxBoundaryData<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg
