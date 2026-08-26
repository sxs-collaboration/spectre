// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "Domain/Structure/DirectionalId.hpp"
#include "Evolution/DgSubcell/InitialTciData.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell::Tags {
/*!
 * \brief Inbox tag for communicating the RDMP and TCI status/decision during
 * initialization.
 *
 * \note The inner map uses `std::unordered_map` rather than
 * `DirectionalIdMap` because elements at non-conforming interfaces (e.g.
 *  inner wedges bordering a spherical shell) can have more than
 * `maximum_number_of_neighbors(Dim)` neighbors sending messages. This inbox
 * is only populated during initialization and is erased after use, so the
 * heap-allocation cost is negligible.
 */
template <size_t Dim>
struct InitialTciData {
  using temporal_id = int;
  using type =
      std::map<temporal_id,
               std::unordered_map<DirectionalId<Dim>,
                                  evolution::dg::subcell::InitialTciData,
                                  boost::hash<DirectionalId<Dim>>>>;

  template <typename ReceiveDataType>
  static bool insert_into_inbox(const gsl::not_null<type*> inbox,
                                const temporal_id& time_step_id,
                                ReceiveDataType&& data) {
    auto& current_inbox = (*inbox)[time_step_id];

    const auto& direction_and_element_id = data.first;

    ASSERT(current_inbox.find(direction_and_element_id) == current_inbox.end(),
           "Received data from direction "
               << direction_and_element_id.direction() << " and element ID "
               << direction_and_element_id.id() << " more than once");
    current_inbox.emplace(std::forward<ReceiveDataType>(data));
    return true;
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
