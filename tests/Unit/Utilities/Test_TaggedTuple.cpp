// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <sstream>

#include "Utilities/StdHelpers.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
struct name {
  using type = std::string;
};

struct age {
  using type = int;
};
struct email {
  using type = std::string;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.TaggedTuple", "[Utilities][Unit]") {
  /// [construction_example]
  TaggedTupleTypelist<typelist<name, age, email>> test("bla", 17,
                                                       "bla@bla.bla");
  /// [construction_example]
  static_assert(TaggedTuple<name, age, email>::size() == 3,
                "Failed to test size of TaggedTuple");
  std::stringstream ss;
  ss << test;
  CHECK(ss.str() == "(bla, 17, bla@bla.bla)");
  CHECK(test.size() == 3);
  /// [get_example]
  CHECK("bla" == test.template get<name>());
  CHECK(17 == test.template get<age>());
  CHECK("bla@bla.bla" == test.template get<email>());
  /// [get_example]
  auto& name_temp = test.template get<name>();
  name_temp = "Dennis";
  CHECK(test.template get<name>() == "Dennis");
  const auto& name_temp2 = test.template get<name>();
  CHECK(name_temp2 == "Dennis");
  auto name_temp3 = std::move(test.template get<name>());
  CHECK(name_temp3 == "Dennis");

  test.template get<name>() = "Eamonn";
  const auto test2 = test;
  CHECK(17 == test2.template get<age>());

  CHECK(test2 == serialize_and_deserialize(test2));
}
