// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <variant>

#include "Framework/TestHelpers.hpp"
#include "Parallel/PupStlCpp17.hpp"
#include "Utilities/Gsl.hpp"

namespace {
struct LocalNoCopyMove {
  LocalNoCopyMove() = default;
  explicit LocalNoCopyMove(const size_t in_t) : t(in_t) {}
  LocalNoCopyMove(const LocalNoCopyMove&) = delete;
  LocalNoCopyMove& operator=(const LocalNoCopyMove&) = delete;
  LocalNoCopyMove(LocalNoCopyMove&&) = delete;
  LocalNoCopyMove& operator=(LocalNoCopyMove&&) = delete;
  ~LocalNoCopyMove() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) { p | t; }

  size_t t;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Serialization.PupStlCpp17", "[Serialization][Unit]") {
  {
    INFO("Optional");
    const std::optional<int> t_not_set{};
    CHECK_FALSE(t_not_set.has_value());
    const auto t_not_set_deserialized = serialize_and_deserialize(t_not_set);
    CHECK_FALSE(t_not_set_deserialized.has_value());

    const std::optional<int> t_set{10};
    REQUIRE(t_set.value() == 10);
    const auto t_set_deserialized = serialize_and_deserialize(t_set);
    REQUIRE(t_set_deserialized.value());
    CHECK(*t_set_deserialized == 10);

    // Test that we can handle a non-copyable and non-movable class. This occurs
    // when you have a hashtable where the mapped type is a
    // std::optional<NonCopyable>
    const std::optional<LocalNoCopyMove> t_nc{10};
    REQUIRE(t_nc.value().t == 10);
    std::optional<LocalNoCopyMove> t_nc_deserialized{};
    serialize_and_deserialize(make_not_null(&t_nc_deserialized), t_nc);
    REQUIRE(t_nc_deserialized.value().t == 10);
    CHECK(t_nc_deserialized->t == 10);
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
