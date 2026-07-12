// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

#include "Domain/Structure/DirectionalId.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::Tags {
template <size_t Dim, bool UseNodegroupDgElements, bool IsAuxiliary>
bool BoundaryCorrectionAndGhostCellsInbox<Dim, UseNodegroupDgElements,
                                          IsAuxiliary>::
    insert_into_inbox(
        const gsl::not_null<type_spsc*> inbox, const temporal_id& time_step_id,
        std::pair<DirectionalId<Dim>, evolution::dg::BoundaryData<Dim>> data) {
  const DirectionalId<Dim>& neighbor_id = data.first;
  // Note: This assumes the neighbor_id is oriented into our (the element
  // whose inbox this is) frame.
  const size_t neighbor_index = inbox->index(neighbor_id);
  if (UNLIKELY(not gsl::at(inbox->boundary_data_in_directions, neighbor_index)
                       .try_emplace(time_step_id, std::move(data.second),
                                    std::move(data.first)))) {
    ERROR(
        "Failed to emplace data into inbox. neighbor_id: ("
        << neighbor_id.direction() << ',' << neighbor_id.id()
        << ") at TimeStepID: " << time_step_id << " the size of the inbox is "
        << gsl::at(inbox->boundary_data_in_directions, neighbor_index).size());
  }
  // Notes:
  // 1. fetch_sub does a post-decrement.
  // 2. We need thread synchronization here, so doing relaxed_order would be a
  //    bug.
  return inbox->missing_messages.fetch_sub(1, std::memory_order_acq_rel) == 1;
}

template <size_t Dim, bool UseNodegroupDgElements, bool IsAuxiliary>
bool BoundaryCorrectionAndGhostCellsInbox<Dim, UseNodegroupDgElements,
                                          IsAuxiliary>::
    insert_into_inbox(
        const gsl::not_null<type_map*> inbox, const temporal_id& time_step_id,
        std::pair<DirectionalId<Dim>, evolution::dg::BoundaryData<Dim>> data) {
  auto& current_inbox = inbox->messages[time_step_id];
  current_inbox.emplace_back(std::move(data));
  --inbox->missing_messages;
  return inbox->missing_messages == 0;
}

template <size_t Dim, bool UseNodegroupDgElements, bool IsAuxiliary>
std::string BoundaryCorrectionAndGhostCellsInbox<
    Dim, UseNodegroupDgElements,
    IsAuxiliary>::output_inbox(const type_spsc& inbox,
                               const size_t padding_size) {
  std::stringstream ss{};
  const std::string pad(padding_size, ' ');
  ss << std::scientific << std::setprecision(16);
  ss << pad << "BoundaryCorrectionAndGhostCellInbox:\n";
  ss << pad
     << "Warning: Printing atomic state is not possible in general so data "
        "printed is limited.\n";
  for (size_t i = 0; i < inbox.boundary_data_in_directions.size(); ++i) {
    const auto& data_in_direction =
        gsl::at(inbox.boundary_data_in_directions, i);
    ss << pad << "Id: "
       << "Approximate size: " << data_in_direction.size() << "\n";
  }

  return ss.str();
}

template <size_t Dim, bool UseNodegroupDgElements, bool IsAuxiliary>
std::string BoundaryCorrectionAndGhostCellsInbox<
    Dim, UseNodegroupDgElements,
    IsAuxiliary>::output_inbox(const type_map& inbox,
                               const size_t padding_size) {
  std::stringstream ss{};
  const std::string pad(padding_size, ' ');
  ss << std::scientific << std::setprecision(16);
  ss << pad << "BoundaryCorrectionAndGhostCellInbox:\n";

  for (const auto& [current_time_step_id, hash_map] : inbox.messages) {
    ss << pad << " Current time: " << current_time_step_id << "\n";
    // We only care about the next time because that's important for deadlock
    // detection. The data itself isn't super important
    for (const auto& [key, boundary_data] : hash_map) {
      ss << pad << "  Key: " << key
         << ", next time: " << boundary_data.validity_range << "\n";
    }
  }

  return ss.str();
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define USE_NODEGROUP_DG_ELEMENTS(data) BOOST_PP_TUPLE_ELEM(1, data)
#define IS_AUXILIARY(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                            \
  template struct BoundaryCorrectionAndGhostCellsInbox< \
      DIM(data), USE_NODEGROUP_DG_ELEMENTS(data), IS_AUXILIARY(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (true, false), (true, false))

#undef INSTANTIATE
#undef IS_AUXILIARY
#undef USE_NODEGROUP_DG_ELEMENTS
#undef DIM
}  // namespace evolution::dg::Tags
