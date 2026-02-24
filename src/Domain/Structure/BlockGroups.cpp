// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/BlockGroups.hpp"

#include <exception>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/StdHelpers.hpp"

namespace domain {

bool block_is_in_group(
    const std::string& block_name, const std::string& block_or_group_name,
    const std::unordered_map<std::string, std::unordered_set<std::string>>&
        block_groups) {
  if (block_name == block_or_group_name) {
    return true;
  }
  const auto block_group_it = block_groups.find(block_or_group_name);
  return block_group_it != block_groups.end() and
         block_group_it->second.count(block_name) == 1;
}

std::unordered_set<std::string> expand_block_groups_to_block_names(
    const std::vector<std::string>& block_or_group_names,
    const std::vector<std::string>& all_block_names,
    const std::unordered_map<std::string, std::unordered_set<std::string>>&
        block_groups) {
  // Use an unordered_set to elide duplicates
  std::unordered_set<std::string> expanded_block_names;
  for (const auto& block_or_group_name : block_or_group_names) {
    if (const auto block_group_it = block_groups.find(block_or_group_name);
        block_group_it != block_groups.end()) {
      expanded_block_names.insert(block_group_it->second.begin(),
                                  block_group_it->second.end());
    } else if (const auto block_name_it =
                   std::find(all_block_names.begin(), all_block_names.end(),
                             block_or_group_name);
               block_name_it != all_block_names.end()) {
      expanded_block_names.insert(*block_name_it);
    } else {
      using ::operator<<;
      ERROR_AS(
          "The block or group '"
              << block_or_group_name
              << "' is not one of the block names or groups of the domain. "
                 "The known block groups are "
              << keys_of(block_groups) << " and the known block names are "
              << all_block_names,
          std::invalid_argument);
    }
  }
  return expanded_block_names;
}

}  // namespace domain
