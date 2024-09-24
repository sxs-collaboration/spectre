// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Domain/Structure/BlockGroups.hpp"

namespace domain {

SPECTRE_TEST_CASE("Unit.Domain.BlockGroups", "[Domain][Unit]") {
  const std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups{
          {"Group1", {"Block1", "Block2"}},
          {"Group2", {"Block3", "Block4"}},
      };
  CHECK(block_is_in_group("Block1", "Group1", block_groups));
  CHECK(block_is_in_group("Block2", "Group1", block_groups));
  CHECK(not block_is_in_group("Block3", "Group1", block_groups));
  CHECK(not block_is_in_group("Block1", "Group2", block_groups));

  const std::vector<std::string> all_block_names{"Block1", "Block2", "Block3",
                                                 "Block4"};
  const std::unordered_set<std::string> expanded_block_names{"Block1", "Block3",
                                                             "Block4"};
  CHECK(expand_block_groups_to_block_names({"Block1", "Group2"},
                                           all_block_names, block_groups) ==
        expanded_block_names);
  CHECK_THROWS_WITH(
      expand_block_groups_to_block_names({"NoBlock", "Group2"}, all_block_names,
                                         block_groups),
      Catch::Matchers::ContainsSubstring(
          "The block or group 'NoBlock' is not one of the block names or "
          "groups of the domain."));
}

}  // namespace domain
