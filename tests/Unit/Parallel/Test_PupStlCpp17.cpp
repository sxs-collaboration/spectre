// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <variant>

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
  {
    INFO("Variant");
    std::variant<int, char, size_t> t{-1};
    CHECK(std::get<int>(serialize_and_deserialize(t)) == -1);
    t = 'a';
    CHECK(std::get<char>(serialize_and_deserialize(t)) == 'a');
    t = 10;
    CHECK(std::get<int>(serialize_and_deserialize(t)) == 10);
    t = static_cast<size_t>(100);
    CHECK(std::get<size_t>(serialize_and_deserialize(t)) == 100);
  }
}
