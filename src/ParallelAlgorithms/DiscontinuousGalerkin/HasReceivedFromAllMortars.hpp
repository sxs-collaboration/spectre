// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Structure/Element.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace dg {

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Determines if data on all mortars has been received for the `InboxTag`
 * at time `temporal_id`.
 *
 * The `InboxTag` must hold a container indexed by the `TemporalIdType`,
 * representing received neighbor data at a given time. Each element in the
 * container, i.e. the received neighbor data at a given time, must be another
 * container indexed by the `dg::MortarId<Dim>`. The value it holds is not
 * relevant for this function. Here's an example for such a type:
 *
 * \snippet Test_HasReceivedFromAllMortars.cpp inbox_example
 */
template <typename InboxTag, size_t Dim, typename TemporalIdType,
          typename... InboxTags>
bool has_received_from_all_mortars(
    const TemporalIdType& temporal_id, const Element<Dim>& element,
    const tuples::TaggedTuple<InboxTags...>& inboxes) noexcept {
  if (UNLIKELY(element.number_of_neighbors() == 0)) {
    return true;
  }
  const auto& inbox = tuples::get<InboxTag>(inboxes);
  const auto temporal_received = inbox.find(temporal_id);
  if (temporal_received == inbox.end()) {
    return false;
  }
  const auto& received_neighbor_data = temporal_received->second;
  for (const auto& direction_and_neighbors : element.neighbors()) {
    const auto& direction = direction_and_neighbors.first;
    for (const auto& neighbor : direction_and_neighbors.second) {
      const auto neighbor_received =
          received_neighbor_data.find(MortarId<Dim>{direction, neighbor});
      if (neighbor_received == received_neighbor_data.end()) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace dg
