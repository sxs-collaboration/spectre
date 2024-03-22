// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>

#include "Parallel/InboxInserters.hpp"

namespace Tags::ChangeSlabSize {
struct NewSlabSizeInbox
    : public Parallel::InboxInserters::MemberInsert<NewSlabSizeInbox> {
  using temporal_id = int64_t;
  using type = std::map<temporal_id, std::unordered_multiset<double>>;

  static std::string output_inbox(const type& inbox,
                                  const size_t padding_size) {
    std::stringstream ss{};
    const std::string pad(padding_size, ' ');

    ss << std::scientific << std::setprecision(16);
    ss << pad << "NewSlabSizeInbox:\n";
    for (const auto& [slab_number, slab_sizes] : inbox) {
      ss << pad << " Slab number: " << slab_number
         << ", slab sizes: " << slab_sizes << "\n";
    }

    return ss.str();
  }
};

// This inbox doesn't receive any data, it just counts messages (using
// the size of the multiset).  Whenever a message is sent to the
// NewSlabSizeInbox, another message is sent here synchronously, so
// the count here is the number of messages we expect in the
// NewSlabSizeInbox.
struct NumberOfExpectedMessagesInbox
    : public Parallel::InboxInserters::MemberInsert<
          NumberOfExpectedMessagesInbox> {
  using temporal_id = int64_t;
  using NoData = std::tuple<>;
  using type = std::map<temporal_id,
                        std::unordered_multiset<NoData, boost::hash<NoData>>>;

  static std::string output_inbox(const type& inbox,
                                  const size_t padding_size) {
    std::stringstream ss{};
    const std::string pad(padding_size, ' ');

    ss << std::scientific << std::setprecision(16);
    ss << pad << "SlabSizeNumberOfExpectedMessagesInbox:\n";
    for (const auto& [number_of_messages, no_data] : inbox) {
      (void)no_data;
      ss << pad << " Number of messages: " << number_of_messages << "\n";
    }

    return ss.str();
  }
};
}  // namespace Tags::ChangeSlabSize
