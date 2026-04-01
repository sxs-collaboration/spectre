// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <vector>

#include "Framework/TestHelpers.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace {
// A type without a default constructor, to test the not_null overload.
struct NoDefault {
  explicit NoDefault(int v) : value(v) {}
  int value;
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) { p | value; }
  bool operator==(const NoDefault& rhs) const { return value == rhs.value; }
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.Serialize", "[Unit][Parallel]") {
  CHECK(size_of_object_in_bytes(std::array<double, 4>{}) == 4 * sizeof(double));
  CHECK(
      size_of_object_in_bytes(std::vector<double>(10)) ==
      (10 * sizeof(double) + sizeof(decltype(std::vector<double>(10).size()))));

  // Test value-returning overload
  {
    const int original = 42;
    const int result = serialize_and_deserialize(original);
    CHECK(result == original);
  }
  {
    const std::vector<double> original{1.0, 2.0, 3.0};
    const auto result = serialize_and_deserialize(original);
    CHECK(result == original);
  }

  // Test not_null overload
  {
    const int original = 7;
    int result = 0;
    serialize_and_deserialize(make_not_null(&result), original);
    CHECK(result == original);
  }
  {
    const std::vector<double> original{4.0, 5.0, 6.0};
    std::vector<double> result{};
    serialize_and_deserialize(make_not_null(&result), original);
    CHECK(result == original);
  }
  // Test not_null overload with a non-default-constructible type
  {
    const NoDefault original{42};
    NoDefault result{0};
    serialize_and_deserialize(make_not_null(&result), original);
    CHECK(result == original);
  }
}
