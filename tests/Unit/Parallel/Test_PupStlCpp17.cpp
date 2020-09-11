// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>

#include "Framework/TestHelpers.hpp"
#include "Parallel/PupStlCpp17.hpp"

SPECTRE_TEST_CASE("Unit.Serialization.PupStlCpp17", "[Serialization][Unit]") {
  {
    INFO("Optional");
    const std::optional<int> t_not_set{};
    CHECK_FALSE(t_not_set.has_value());
    const auto t_not_set_deserialized = serialize_and_deserialize(t_not_set);
    CHECK_FALSE(t_not_set_deserialized.has_value());

    const std::optional<int> t_set{10};
    REQUIRE(t_set.value());
    const auto t_set_deserialized = serialize_and_deserialize(t_set);
    REQUIRE(t_set_deserialized.value());
    CHECK(*t_set_deserialized == 10);
  }
}
