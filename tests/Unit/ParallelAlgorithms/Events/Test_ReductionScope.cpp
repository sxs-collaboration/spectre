// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Framework/TestCreation.hpp"
#include "ParallelAlgorithms/Events/ReductionScope.hpp"
#include "Utilities/GetOutput.hpp"

namespace {
void test_creation() {
  for (const auto reduction_scope :
       {Events::ReductionScope::WholeDomain, Events::ReductionScope::ByBlock,
        Events::ReductionScope::ByBlockOrGroup}) {
    CHECK(TestHelpers::test_creation<Events::ReductionScope>(
              get_output(reduction_scope)) == reduction_scope);
  }
  CHECK(get_output(Events::ReductionScope::WholeDomain) == "WholeDomain");
  CHECK(get_output(Events::ReductionScope::ByBlock) == "ByBlock");
  CHECK(get_output(Events::ReductionScope::ByBlockOrGroup) == "ByBlockOrGroup");
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<Events::ReductionScope>("PerBlock"),
      Catch::Matchers::ContainsSubstring(
          "Failed to convert \"PerBlock\" to ReductionScope"));
}

void test_reduction_subfile_suffix() {
  const std::vector<std::string> all_block_names{"Shell0", "Shell1", "Cube"};
  const std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups{{"Shells", {"Shell0", "Shell1"}}};
  const auto suffix =
      [&all_block_names, &block_groups](
          const Events::ReductionScope reduction_scope,
          const std::optional<std::vector<std::string>>& blocks_to_observe,
          const std::string& block_name) {
        return Events::reduction_subfile_suffix(reduction_scope,
                                                blocks_to_observe, block_name,
                                                all_block_names, block_groups);
      };
  using Events::ReductionScope;
  const std::optional<std::string> none{};
  const std::optional<std::vector<std::string>> all{};
  const std::vector<std::string> shells{"Shells"};
  const std::vector<std::string> shells_and_shell0{"Shells", "Shell0"};

  // All blocks
  CHECK(suffix(ReductionScope::WholeDomain, all, "Shell0") == "");
  CHECK(suffix(ReductionScope::ByBlock, all, "Shell0") == "/Shell0");
  CHECK(suffix(ReductionScope::ByBlockOrGroup, all, "Cube") == "/Cube");

  // Restricted to a group
  CHECK(suffix(ReductionScope::WholeDomain, shells, "Shell1") == "");
  CHECK(suffix(ReductionScope::WholeDomain, shells, "Cube") == none);
  CHECK(suffix(ReductionScope::ByBlock, shells, "Shell1") == "/Shell1");
  CHECK(suffix(ReductionScope::ByBlock, shells, "Cube") == none);
  CHECK(suffix(ReductionScope::ByBlockOrGroup, shells, "Shell1") == "/Shells");
  CHECK(suffix(ReductionScope::ByBlockOrGroup, shells, "Cube") == none);

  // Overlapping entries are fine when reducing by individual block
  CHECK(suffix(ReductionScope::ByBlock, shells_and_shell0, "Shell0") ==
        "/Shell0");
  CHECK_THROWS_WITH(
      suffix(ReductionScope::ByBlockOrGroup, shells_and_shell0, "Shell0"),
      Catch::Matchers::ContainsSubstring(
          "With ReductionScope 'ByBlockOrGroup' each block can be in at most "
          "one entry of BlocksToObserve"));
  CHECK_THROWS_WITH(
      suffix(ReductionScope::WholeDomain, std::vector<std::string>{"NoBlock"},
             "Shell0"),
      Catch::Matchers::ContainsSubstring(
          "The block or group 'NoBlock' is not one of the block names or "
          "groups of the domain."));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Events.ReductionScope",
                  "[Unit][ParallelAlgorithms]") {
  test_creation();
  test_reduction_subfile_suffix();
}
