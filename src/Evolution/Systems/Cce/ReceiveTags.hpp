// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iomanip>
#include <sstream>
#include <string>

#include "DataStructures/Variables.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Time/TimeStepId.hpp"

namespace Cce {
/// \brief Tags used by CCE for communication.
namespace ReceiveTags {

/// A receive tag for the data sent to the CCE evolution component from the CCE
/// boundary component
template <typename CommunicationTagList>
struct BoundaryData
    : Parallel::InboxInserters::Value<BoundaryData<CommunicationTagList>> {
  using temporal_id = TimeStepId;
  using type = std::unordered_map<temporal_id, Variables<CommunicationTagList>>;

  static std::string output_inbox(const type& inbox,
                                  const size_t padding_size) {
    std::stringstream ss{};
    const std::string pad(padding_size, ' ');

    ss << std::scientific << std::setprecision(16);
    ss << pad << "CceBoundaryDataInbox:\n";
    // We don't really care about the variables, just the times
    for (const auto& [current_time_step_id, variables] : inbox) {
      (void)variables;
      ss << pad << " Time: " << current_time_step_id << "\n";
    }

    return ss.str();
  }
};

}  // namespace ReceiveTags
}  // namespace Cce
