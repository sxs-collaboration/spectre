// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Events/ReductionScope.hpp"

#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Domain/Structure/BlockGroups.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/StdHelpers.hpp"

namespace Events {
std::ostream& operator<<(std::ostream& os,
                         const ReductionScope reduction_scope) {
  switch (reduction_scope) {
    case ReductionScope::WholeDomain:
      return os << "WholeDomain";
    case ReductionScope::ByBlock:
      return os << "ByBlock";
    case ReductionScope::ByBlockOrGroup:
      return os << "ByBlockOrGroup";
    default:
      ERROR("Unknown ReductionScope "
            << static_cast<int>(reduction_scope)
            << ". Must be WholeDomain, ByBlock, or ByBlockOrGroup.");
  }
}

std::optional<std::string> block_subfile_suffix(
    const ReductionScope reduction_scope,
    const std::optional<std::vector<std::string>>& blocks_to_observe,
    const std::string& block_name,
    const std::vector<std::string>& all_block_names,
    const std::unordered_map<std::string, std::unordered_set<std::string>>&
        block_groups) {
  if (not blocks_to_observe.has_value()) {
    if (reduction_scope == ReductionScope::WholeDomain) {
      return std::string{};
    }
    // Both ByBlock and ByBlockOrGroup reduce per block when observing all
    // blocks
    return "/" + block_name;
  }
  const std::vector<std::string> containing_names =
      domain::block_or_group_names_containing_block(
          block_name, blocks_to_observe.value(), all_block_names, block_groups);
  if (containing_names.empty()) {
    return std::nullopt;
  }
  switch (reduction_scope) {
    case ReductionScope::WholeDomain:
      return std::string{};
    case ReductionScope::ByBlock:
      return "/" + block_name;
    case ReductionScope::ByBlockOrGroup:
      if (containing_names.size() > 1) {
        using ::operator<<;
        ERROR(
            "With ReductionScope 'ByBlockOrGroup' each block can be in at "
            "most one entry of BlocksToObserve, because each element "
            "contributes to only one reduction. However, the block '"
            << block_name << "' is in the entries " << containing_names
            << " of BlocksToObserve " << blocks_to_observe.value()
            << ". Remove the overlapping entries, use ReductionScope "
               "'ByBlock', or add a separate event.");
      }
      return "/" + containing_names.front();
    default:
      ERROR("Unknown ReductionScope " << static_cast<int>(reduction_scope));
  }
}
}  // namespace Events

template <>
Events::ReductionScope
Options::create_from_yaml<Events::ReductionScope>::create<void>(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  for (const auto reduction_scope :
       {Events::ReductionScope::WholeDomain, Events::ReductionScope::ByBlock,
        Events::ReductionScope::ByBlockOrGroup}) {
    if (type_read == get_output(reduction_scope)) {
      return reduction_scope;
    }
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read
                  << "\" to ReductionScope. Must be one of 'WholeDomain', "
                     "'ByBlock', or 'ByBlockOrGroup'.");
}
