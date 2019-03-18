// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"

namespace {
struct Tag : db::SimpleTag {
  static std::string name() noexcept { return "Tag"; }
  using type = double;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.Prefixes",
                  "[Unit][DataStructures]") {
  /// [source_name]
  CHECK(Tags::Source<Tag>::name() == "Source(" + Tag::name() + ")");
  /// [source_name]
  /// [next_name]
  CHECK(Tags::Next<Tag>::name() == "Next(" + Tag::name() + ")");
  /// [next_name]
  CHECK(Tags::dt<Tag>::name() == "dt(Tag)");
}
