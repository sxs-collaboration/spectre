// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "IO/Observer/Tags.hpp"

namespace observers {

/// Retrieve the `observers::Tags::ObservationKey<SectionIdTag>`, or the empty
/// string if `SectionIdTag` is void. This is useful to support sections in
/// parallel algorithms. The return value can be used to construct a subfile
/// path for observations, and to skip observations on elements that are not
/// part of a section (`std::nullopt`).
template <typename SectionIdTag, typename DbTagsList>
std::optional<std::string> get_section_observation_key(
    [[maybe_unused]] const db::DataBox<DbTagsList>& box) noexcept {
  if constexpr (std::is_same_v<SectionIdTag, void>) {
    return "";
  } else {
    return db::get<observers::Tags::ObservationKey<SectionIdTag>>(box);
  }
}

}  // namespace observers
