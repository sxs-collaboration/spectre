// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
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
  using type =
      std::map<temporal_id,
               DirectionalIdMap<Dim, evolution::dg::subcell::InitialTciData>>;

  template <typename ReceiveDataType>
  static void insert_into_inbox(const gsl::not_null<type*> inbox,
                                const temporal_id& time_step_id,
                                ReceiveDataType&& data) {
    auto& current_inbox = (*inbox)[time_step_id];

    const auto& direction_and_element_id = data.first;

    ASSERT(current_inbox.find(direction_and_element_id) == current_inbox.end(),
           "Received data from direction "
               << direction_and_element_id.direction() << " and element ID "
               << direction_and_element_id.id() << " more than once");
    current_inbox.emplace(std::forward<ReceiveDataType>(data));
  }

  static std::string output_inbox(const type& inbox,
                                  const size_t padding_size) {
    std::stringstream ss{};
    const std::string pad(padding_size, ' ');

    ss << std::scientific << std::setprecision(16);
    ss << pad << "InitialTciDataInbox:\n";
    for (const auto& [action_number, hash_map] : inbox) {
      ss << pad << " Action number: " << action_number << "\n";
      for (const auto& [key, initial_tci_data] : hash_map) {
        using ::operator<<;
        ss << pad << "  Key: " << key
           << ", TCI: " << initial_tci_data.tci_status << "\n";
      }
    }

    return ss.str();
  }
};
}  // namespace evolution::dg::subcell::Tags
