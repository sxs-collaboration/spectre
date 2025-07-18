// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestTags.hpp"

namespace {
struct NamedBase : db::BaseTag {
  static std::string name() { return "NamedBaseName"; }
};

struct NamedSimple : db::SimpleTag {
  static std::string name() { return "NamedSimpleName"; }
  using type = int;
};

struct SimpleFromOption : TestHelpers::db::Tags::Simple {
  using base = TestHelpers::db::Tags::Simple;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.TestHelpers",
                  "[Unit][DataStructures]") {
  // We can't check most failure cases, so we just check that some
  // valid cases are accepted.
  TestHelpers::db::test_base_tag<TestHelpers::db::Tags::Base>("Base");
  TestHelpers::db::test_base_tag<NamedBase>("NamedBaseName");
  TestHelpers::db::test_simple_tag<TestHelpers::db::Tags::Simple>("Simple");
  TestHelpers::db::test_simple_tag<NamedSimple>("NamedSimpleName");
  TestHelpers::db::test_simple_tag<TestHelpers::db::Tags::SimpleWithBase>(
      "SimpleWithBase");
  TestHelpers::db::test_simple_tag<SimpleFromOption>("Simple");
  TestHelpers::db::test_compute_tag<TestHelpers::db::Tags::SimpleCompute>(
      "Simple");
}
