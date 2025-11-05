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
template <size_t Dim, bool UseNodegroupDgElements>
bool BoundaryCorrectionAndGhostCellsInbox<Dim, UseNodegroupDgElements>::
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
    ERROR("Failed to emplace data into inbox. neighbor_id: ("
          << neighbor_id.direction() << ',' << neighbor_id.id()
          << ") at TimeStepID: " << time_step_id << " the size of the inbox is "
          << gsl::at(inbox->boundary_data_in_directions, neighbor_index).size()
          << " the message count is " << inbox->message_count.load()
          << " and the number of neighbors is "
          << inbox->number_of_neighbors.load());
  }
  // Notes:
  // 1. fetch_add does a post-increment.
  // 2. We need thread synchronization here, so doing relaxed_order would be a
  //    bug.
  // inbox->message_count.fetch_add(1, std::memory_order_acq_rel) + 1;
  return true;
}

template <size_t Dim, bool UseNodegroupDgElements>
bool BoundaryCorrectionAndGhostCellsInbox<Dim, UseNodegroupDgElements>::
    insert_into_inbox(
        const gsl::not_null<type_map*> inbox, const temporal_id& time_step_id,
        std::pair<DirectionalId<Dim>, evolution::dg::BoundaryData<Dim>> data) {
  auto& current_inbox = (*inbox)[time_step_id];
  if (auto it = current_inbox.find(data.first); it != current_inbox.end()) {
    merge_boundary_data(make_not_null(&it->second), std::move(data.second));
  } else {
    // We have not received ghost cells or fluxes at this time.
    if (not current_inbox.insert(std::move(data)).second) {
      ERROR("Failed to insert data to receive at instance '"
            << time_step_id
            << "' with tag 'BoundaryCorrectionAndGhostCellsInbox'.\n");
    }
  }
  return true;
}

template <size_t Dim, bool UseNodegroupDgElements>
std::string
BoundaryCorrectionAndGhostCellsInbox<Dim, UseNodegroupDgElements>::output_inbox(
    const type_spsc& inbox, const size_t padding_size) {
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

template <size_t Dim, bool UseNodegroupDgElements>
std::string
BoundaryCorrectionAndGhostCellsInbox<Dim, UseNodegroupDgElements>::output_inbox(
    const type_map& inbox, const size_t padding_size) {
  std::stringstream ss{};
  const std::string pad(padding_size, ' ');
  ss << std::scientific << std::setprecision(16);
  ss << pad << "BoundaryCorrectionAndGhostCellInbox:\n";

  for (const auto& [current_time_step_id, hash_map] : inbox) {
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

#define INSTANTIATE(_, data)                            \
  template struct BoundaryCorrectionAndGhostCellsInbox< \
      DIM(data), USE_NODEGROUP_DG_ELEMENTS(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (true, false))

#undef INSTANTIATE
#undef USE_NODEGROUP_DG_ELEMENTS
#undef DIM
}  // namespace evolution::dg::Tags
