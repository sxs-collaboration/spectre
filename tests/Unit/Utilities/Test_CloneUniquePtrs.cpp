// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <memory>
#include <unordered_map>

#include "Utilities/CloneUniquePtrs.hpp"

namespace {
struct NonCopyableValue {
  explicit constexpr NonCopyableValue(const int value_in) : value(value_in) {}
  constexpr NonCopyableValue(const NonCopyableValue&) = delete;
  constexpr NonCopyableValue& operator=(const NonCopyableValue&) = delete;
  constexpr NonCopyableValue(NonCopyableValue&&) = default;
  NonCopyableValue& operator=(NonCopyableValue&&) = default;
  ~NonCopyableValue() = default;

  std::unique_ptr<NonCopyableValue> get_clone() {
    return std::make_unique<NonCopyableValue>(value);
  }

  int value;
};

void test_unordered_map() {
  using Map = std::unordered_map<int, std::unique_ptr<NonCopyableValue>>;
  const auto check_cloning = [](const Map& map) noexcept {
    const auto map_copy = clone_unique_ptrs(map);
    REQUIRE(map.size() == map_copy.size());
    for (const auto& kv : map) {
      CHECK(map_copy.at(kv.first)->value == kv.second->value);
    }
  };
  check_cloning({});

  Map t0{};
  t0[10] = std::make_unique<NonCopyableValue>(150);
  check_cloning(t0);

  Map t1{};
  t1[10] = std::make_unique<NonCopyableValue>(150);
  t1[32] = std::make_unique<NonCopyableValue>(1);
  check_cloning(t1);

  Map t2{};
  t2[10] = std::make_unique<NonCopyableValue>(150);
  t2[32] = std::make_unique<NonCopyableValue>(1);
  t2[-1] = std::make_unique<NonCopyableValue>(100);
  check_cloning(t2);
}

void test_vector() {
  using Vector = std::vector<std::unique_ptr<NonCopyableValue>>;
  const auto check_cloning = [](const Vector& vector) noexcept {
    const auto vector_copy = clone_unique_ptrs(vector);
    REQUIRE(vector.size() == vector_copy.size());
    for (size_t i = 0; i < vector.size(); ++i) {
      CHECK(vector_copy[i]->value == vector[i]->value);
    }
  };

  check_cloning({});

  Vector t0{1};
  t0[0] = std::make_unique<NonCopyableValue>(10);
  check_cloning(t0);

  Vector t1{2};
  t1[0] = std::make_unique<NonCopyableValue>(10);
  t1[1] = std::make_unique<NonCopyableValue>(100);
  check_cloning(t1);

  Vector t2{3};
  t2[0] = std::make_unique<NonCopyableValue>(10);
  t2[1] = std::make_unique<NonCopyableValue>(100);
  t2[2] = std::make_unique<NonCopyableValue>(200);
  check_cloning(t2);
}

SPECTRE_TEST_CASE("Unit.Utilities.CloneUniquePtrs", "[Unit][Utilities]") {
  test_unordered_map();
  test_vector();
}
}  // namespace
