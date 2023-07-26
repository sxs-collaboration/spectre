// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <map>
#include <utility>

#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/DgSubcell/InitialTciData.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell::Tags {
/*!
 * \brief Inbox tag for communicating the RDMP and TCI status/decision during
 * initialization.
 */
template <size_t Dim>
struct InitialTciData {
  using temporal_id = int;
  using type = std::map<
      temporal_id,
      FixedHashMap<maximum_number_of_neighbors(Dim),
                   std::pair<Direction<Dim>, ElementId<Dim>>,
                   evolution::dg::subcell::InitialTciData,
                   boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>>;

  template <typename ReceiveDataType>
  static void insert_into_inbox(const gsl::not_null<type*> inbox,
                                const temporal_id& time_step_id,
                                ReceiveDataType&& data) {
    auto& current_inbox = (*inbox)[time_step_id];

    const auto& direction_and_element_id = data.first;

    ASSERT(current_inbox.find(direction_and_element_id) == current_inbox.end(),
           "Received data from direction "
               << direction_and_element_id.first << " and element ID "
               << direction_and_element_id.second << " more than once");
    current_inbox.emplace(std::forward<ReceiveDataType>(data));
  }
};
}  // namespace evolution::dg::subcell::Tags
