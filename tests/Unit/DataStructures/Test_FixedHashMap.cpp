// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <ostream>
#include <pup.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Direction.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/StdHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
template <size_t Dim>
void check_single_element_map_with_direction(
    const Direction<Dim>& direction) noexcept {
  FixedHashMap<2 * Dim, Direction<Dim>, std::pair<double, NonCopyable>,
               DirectionHash<Dim>>
      map;

  map.emplace(direction, std::make_pair(1.5, NonCopyable{}));
  const typename decltype(map)::value_type entry(
      cpp17::as_const(direction), std::make_pair(1.5, NonCopyable{}));

  const auto& const_map = map;

  test_iterators(map);
  check_cmp(const_map.begin(), const_map.end());
  check_cmp(map.begin(), map.end());
  check_cmp(map.cbegin(), map.end());
  check_cmp(map.begin(), map.cend());

  CHECK(not const_map.empty());
  CHECK(const_map.size() == 1);

  CHECK(const_map.count(direction) == 1);
  CHECK(const_map.count(direction.opposite()) == 0);

  CHECK(const_map.find(direction) == const_map.begin());
  CHECK(const_map.find(direction.opposite()) == const_map.end());

  CHECK(*const_map.begin() == entry);
  CHECK(&const_map.at(direction) == &const_map.begin()->second);

  auto it = map.begin();
  CHECK(*it == entry);
  CHECK(it.operator->() == &*it);
  auto it2 = it;
  CHECK(*it++ == entry);
  CHECK(it != it2);
  CHECK(*it2 == entry);
  CHECK(it == map.end());
  CHECK(++it2 == map.end());
  CHECK(it2 == map.end());
}

template <size_t MaxSize>
void test_direction_key() {
  for (const auto& direction : Direction<2>::all_directions()) {
    check_single_element_map_with_direction(direction);
  }

  FixedHashMap<MaxSize, Direction<2>, double, DirectionHash<2>> map;

  CHECK(map.empty());
  CHECK(map.size() == 0);  // NOLINT(readability-container-size-empty)

  CHECK(map.begin() == cpp17::as_const(map).begin());
  CHECK(map.begin() == cpp17::as_const(map).cbegin());
  CHECK(map.end() == cpp17::as_const(map).end());
  CHECK(map.end() == cpp17::as_const(map).cend());
  CHECK(map.begin() == map.end());

  const auto dir1 = Direction<2>::lower_eta();
  const auto dir2 = Direction<2>::upper_xi();
  const auto dir3 = Direction<2>::lower_xi();
  {
    CHECK(map.count(dir1) == 0);
    const auto result = map.emplace(dir1, 1.);
    CHECK(result.second);
    CHECK(result.first == map.find(dir1));
    CHECK(map.count(dir1) == 1);
  }
  CHECK(not map.empty());
  CHECK(map.size() == 1);
  {
    CHECK(map.count(dir2) == 0);
    const auto result = map.insert(std::make_pair(dir2, 2.));
    CHECK(result.second);
    CHECK(result.first == map.find(dir2));
    CHECK(map.count(dir2) == 1);
  }
  CHECK(not map.empty());
  CHECK(map.size() == 2);
  {
    const auto result = map.insert(std::make_pair(dir2, 3.));
    CHECK(not result.second);
    CHECK(result.first == map.find(dir2));
    CHECK(map.count(dir2) == 1);
  }
  CHECK(map.size() == 2);
  {
    const auto result = map.emplace(dir1, 4.);
    CHECK(not result.second);
    CHECK(result.first == map.find(dir1));
    CHECK(map.count(dir1) == 1);
  }
  CHECK(map.size() == 2);
  CHECK(cpp17::as_const(map).at(dir1) == 1.);
  CHECK(cpp17::as_const(map).at(dir2) == 2.);
  map.at(dir2) = 5.;
  CHECK(cpp17::as_const(map).at(dir2) == 5.);
  CHECK(map.size() == 2);
  CHECK(&cpp17::as_const(map).find(dir1)->second ==
        &cpp17::as_const(map).at(dir1));
  CHECK(&cpp17::as_const(map).find(dir2)->second ==
        &cpp17::as_const(map).at(dir2));
  CHECK(cpp17::as_const(map).find(dir3) == map.end());

  const auto check_exception = [&dir3](auto& passed_map) noexcept {
    try {
      passed_map.at(dir3);
      CHECK(false);
    } catch (const std::out_of_range& e) {
      CHECK(e.what() == get_output(dir3) + " not in FixedHashMap");
    } catch (...) {
      CHECK(false);
    }
  };
  check_exception(map);
  check_exception(cpp17::as_const(map));

  map[dir3] = 6.;
  CHECK(map.size() == 3);
  CHECK(cpp17::as_const(map).at(dir3) == 6.);

  test_iterators(map);

  {
    // The map doesn't guarantee an iteration order, so this block
    // would put the map into a difficult-to-describe state.  We use a
    // copy to not have to deal with that.
    auto map2 = map;
    const auto second = std::next(map2.begin());
    const auto third = std::next(map2.begin(), 2);
    // Checks iterator mutability correctness for erase
    const auto next = map2.erase(cpp17::as_const(map2).find(second->first));
    CHECK(map2.size() == 2);
    CHECK(next == third);
    next->second = -1.;
    CHECK(std::next(map2.begin())->second == -1.);
  }

  CHECK(map == map);
  CHECK_FALSE(map != map);
  {
    // Test copy and move
    const auto check_equal = [](const auto& a, const auto& b) noexcept {
      CHECK(a == b);
      CHECK(b == a);
      CHECK_FALSE(a != b);
      CHECK_FALSE(b != a);
    };
    auto map2 = map;
    check_equal(map, map2);
    auto map3 = std::move(map2);
    check_equal(map, map3);
    decltype(map) map4;
    map4 = std::move(map3);
    check_equal(map, map4);
    decltype(map) map5;
    map5 = map4;
    check_equal(map, map5);

    // Test moving non-copyable entries
    FixedHashMap<MaxSize, Direction<2>, NonCopyable, DirectionHash<2>> nc_map;
    nc_map.emplace(Direction<2>::upper_xi(), NonCopyable{});
    FixedHashMap<MaxSize, Direction<2>, NonCopyable, DirectionHash<2>> nc_map2;
    nc_map2.emplace(Direction<2>::upper_xi(), NonCopyable{});

    check_equal(nc_map, nc_map2);
    auto nc_map3 = std::move(nc_map2);
    check_equal(nc_map, nc_map3);
    FixedHashMap<MaxSize, Direction<2>, NonCopyable, DirectionHash<2>> nc_map4;
    nc_map4 = std::move(nc_map3);
    check_equal(nc_map, nc_map4);
  }
  {
    auto map2 = map;
    map2.erase(dir1);
    CHECK_FALSE(map == map2);
    CHECK_FALSE(map2 == map);
    CHECK(map != map2);
    CHECK(map2 != map);
  }
  {
    auto map2 = map;
    map2.at(dir1) = -1.;
    CHECK_FALSE(map == map2);
    CHECK_FALSE(map2 == map);
    CHECK(map != map2);
    CHECK(map2 != map);
  }
  {
    // Check initializer list constructor
    CHECK(decltype(map)(
              std::initializer_list<typename decltype(map)::value_type>{})
              .empty());
    CHECK(map == decltype(map){{dir1, map.at(dir1)},
                               {dir2, map.at(dir2)},
                               {dir3, map.at(dir3)}});
    CHECK(map == decltype(map){{dir2, map.at(dir2)},
                               {dir3, map.at(dir3)},
                               {dir1, map.at(dir1)}});
  }

  test_serialization(map);

  CHECK(get_output(map) == get_output(std::unordered_map<Direction<2>, double>(
                               map.begin(), map.end())));

  CHECK(map.contains(dir1));
  CHECK(map.erase(dir1) == 1);
  CHECK_FALSE(map.contains(dir1));
  CHECK(map.size() == 2);
  CHECK(map.find(dir1) == map.end());
  CHECK(map.erase(dir1) == 0);
  CHECK(map.size() == 2);
  CHECK(map.find(dir1) == map.end());

  map.clear();
  CHECK(map.empty());
  CHECK(map.size() == 0);  // NOLINT(readability-container-size-empty)
  test_iterators(map);

  test_copy_semantics(map);
  const auto map2 = map;
  test_move_semantics(std::move(map), map2);
}

// A non-copyable size_t wrapper for testing non-copyable keys.
struct NonCopyableSizeT {
  constexpr NonCopyableSizeT() = default;
  constexpr NonCopyableSizeT(const NonCopyableSizeT&) = delete;
  constexpr NonCopyableSizeT& operator=(const NonCopyableSizeT&) = delete;
  constexpr NonCopyableSizeT(NonCopyableSizeT&&) = default;
  NonCopyableSizeT& operator=(NonCopyableSizeT&&) = default;
  ~NonCopyableSizeT() = default;

  // Intentional implicit conversion
  // NOLINTNEXTLINE(google-explicit-constructor)
  NonCopyableSizeT(size_t t) : value(t) {}

  void pup(PUP::er& p) noexcept { p | value; }  // NOLINT

  size_t value = 0;
};
inline bool operator==(const NonCopyableSizeT& a,
                       const NonCopyableSizeT& b) noexcept {
  return a.value == b.value;
}
inline bool operator!=(const NonCopyableSizeT& a,
                       const NonCopyableSizeT& b) noexcept {
  return not(a == b);
}
inline std::ostream& operator<<(std::ostream& os,
                                const NonCopyableSizeT& v) noexcept {
  return os << v.value;
}

// A hash that hashes two inputs to the same value
struct RepeatedHash {
  size_t operator()(const size_t t) const noexcept {
    if (t == 3) {
      return 2;
    }
    if (t == 4) {
      return 5;
    }
    return t;
  }

  size_t operator()(const NonCopyableSizeT& t) const noexcept {
    if (t.value == 3) {
      return 2;
    }
    if (t.value == 4) {
      return 5;
    }
    return t.value;
  }
};

template <typename KeyType>
// NOLINTNEXTLINE(google-readability-function-size)
void test_repeated_key() {
  using CopyableKeyMapType = FixedHashMap<6, size_t, size_t, RepeatedHash>;
  using NonCopyableKeyMapType =
      FixedHashMap<6, NonCopyableSizeT, size_t, RepeatedHash>;
  FixedHashMap<6, KeyType, size_t, RepeatedHash> map;
  CHECK(map.empty());
  // NOLINTNEXTLINE(readability-container-size-empty)
  CHECK(map.size() == 0);

  map.emplace(1, 11);
  CHECK_FALSE(map.empty());
  CHECK(map.size() == 1);
  CHECK(map.at(1) == 11);
  CHECK(map.count(1) == 1);
  CHECK(map.count(2) == 0);
  CHECK(map.contains(1));
  CHECK_FALSE(map.contains(2));
  CHECK_FALSE(map.contains(3));
  CHECK(map.find(1) != map.end());
  CHECK(map.find(2) == map.end());
  CHECK(cpp17::as_const(map).find(1) != map.end());
  CHECK(cpp17::as_const(map).find(2) == map.end());

  make_overloader([](gsl::not_null<CopyableKeyMapType*>
                         local_map) noexcept { (*local_map)[2] = 12; },
                  [](gsl::not_null<NonCopyableKeyMapType*> local_map) noexcept {
                    local_map->insert({2, 12});
                  })(make_not_null(&map));
  CHECK(map.size() == 2);
  CHECK_FALSE(map.empty());
  CHECK(map.at(2) == 12);
  CHECK(map.count(1) == 1);
  CHECK(map.count(2) == 1);
  CHECK(map.count(3) == 0);
  CHECK(map.contains(1));
  CHECK(map.contains(2));
  CHECK_FALSE(map.contains(3));
  CHECK(map.find(1) != map.end());
  CHECK(map.find(2) != map.end());
  CHECK(map.find(3) == map.end());
  CHECK(cpp17::as_const(map).find(1) != map.end());
  CHECK(cpp17::as_const(map).find(2) != map.end());
  CHECK(cpp17::as_const(map).find(3) == map.end());

  map.insert({3, 13});
  CHECK(map.size() == 3);
  CHECK_FALSE(map.empty());
  CHECK(map.at(3) == 13);
  CHECK(cpp17::as_const(map).at(3) == 13);
  make_overloader(
      [](gsl::not_null<CopyableKeyMapType*> local_map) noexcept {
        CHECK((*local_map)[3] == 13);
      },
      [](gsl::not_null<NonCopyableKeyMapType*> /*local_map*/) noexcept {})(
      make_not_null(&map));
  CHECK(map.count(1) == 1);
  CHECK(map.count(2) == 1);
  CHECK(map.count(3) == 1);
  CHECK(map.contains(1));
  CHECK(map.contains(2));
  CHECK(map.contains(3));
  CHECK(map.find(1) != map.end());
  CHECK(map.find(2) != map.end());
  CHECK(map.find(3) != map.end());
  CHECK(map.find(2) != map.find(3));
  CHECK(cpp17::as_const(map).find(1) != map.end());
  CHECK(cpp17::as_const(map).find(2) != map.end());
  CHECK(cpp17::as_const(map).find(3) != map.end());
  CHECK(cpp17::as_const(map).find(2) != map.find(3));

  map.erase(2);
  CHECK(map.size() == 2);
  CHECK_FALSE(map.empty());
  CHECK(map.count(1) == 1);
  CHECK(map.count(2) == 0);
  CHECK(map.count(3) == 1);
  CHECK(map.contains(1));
  CHECK_FALSE(map.contains(2));
  CHECK(map.contains(3));
  CHECK(map.find(1) != map.end());
  CHECK(map.find(2) == map.end());
  CHECK(map.find(3) != map.end());
  CHECK(map.find(2) != map.find(3));

  // Now check key that results in wrapping past end of storage_type
  map.insert({4, 14});
  CHECK(map.size() == 3);
  CHECK_FALSE(map.empty());
  CHECK(map.count(1) == 1);
  CHECK(map.count(2) == 0);
  CHECK(map.count(3) == 1);
  CHECK(map.count(4) == 1);
  CHECK(map.contains(4));
  CHECK_FALSE(map.contains(2));
  CHECK(map.find(1) != map.end());
  CHECK(map.find(2) == map.end());
  CHECK(map.find(3) != map.end());
  CHECK(map.find(4) != map.end());
  CHECK(map.at(4) == 14);
  CHECK(cpp17::as_const(map).at(4) == 14);
  make_overloader(
      [](gsl::not_null<CopyableKeyMapType*> local_map) noexcept {
        CHECK((*local_map)[4] == 14);
      },
      [](gsl::not_null<NonCopyableKeyMapType*> /*local_map*/) noexcept {})(
      make_not_null(&map));

  make_overloader([](gsl::not_null<CopyableKeyMapType*>
                         local_map) noexcept { (*local_map)[5] = 15; },
                  [](gsl::not_null<NonCopyableKeyMapType*> local_map) noexcept {
                    local_map->insert_or_assign(5, 15);
                  })(make_not_null(&map));
  CHECK(map.size() == 4);
  CHECK_FALSE(map.empty());
  CHECK(map.count(1) == 1);
  CHECK(map.count(2) == 0);
  CHECK(map.count(3) == 1);
  CHECK(map.count(4) == 1);
  CHECK(map.count(5) == 1);
  CHECK(map.contains(5));
  CHECK(map.find(1) != map.end());
  CHECK(map.find(2) == map.end());
  CHECK(map.find(3) != map.end());
  CHECK(map.find(4) != map.end());
  CHECK(map.find(5) != map.end());
  CHECK(map.find(4) != map.find(5));
  CHECK(map.at(5) == 15);
  CHECK(cpp17::as_const(map).at(5) == 15);
  make_overloader(
      [](gsl::not_null<CopyableKeyMapType*> local_map) noexcept {
        CHECK((*local_map)[5] == 15);
      },
      [](gsl::not_null<NonCopyableKeyMapType*> /*local_map*/) noexcept {})(
      make_not_null(&map));

  map.erase(4);
  CHECK(map.size() == 3);
  CHECK_FALSE(map.empty());
  CHECK(map.count(1) == 1);
  CHECK(map.count(2) == 0);
  CHECK(map.count(3) == 1);
  CHECK(map.count(5) == 1);
  CHECK(map.contains(5));
  CHECK(map.find(1) != map.end());
  CHECK(map.find(2) == map.end());
  CHECK(map.find(3) != map.end());
  CHECK(map.find(4) == map.end());
  CHECK(map.find(5) != map.end());
  CHECK(map.find(4) != map.find(5));
  CHECK(map.at(5) == 15);
  CHECK(cpp17::as_const(map).at(5) == 15);
  make_overloader(
      [](gsl::not_null<CopyableKeyMapType*> local_map) noexcept {
        CHECK((*local_map)[5] == 15);
      },
      [](gsl::not_null<NonCopyableKeyMapType*> /*local_map*/) noexcept {})(
      make_not_null(&map));

  const auto check_4 = [](
      const auto& l_it_bool_to_4, const size_t expected, const bool inserted,
      // GCC warns about shadowing so we make a very specific name for the map
      auto check_4_local_map) noexcept {
    CHECK(l_it_bool_to_4.second == inserted);
    CHECK(check_4_local_map->size() == 4);
    CHECK_FALSE(check_4_local_map->empty());
    CHECK(check_4_local_map->count(1) == 1);
    CHECK(check_4_local_map->count(2) == 0);
    CHECK(check_4_local_map->count(3) == 1);
    CHECK(check_4_local_map->count(4) == 1);
    CHECK(check_4_local_map->count(5) == 1);
    CHECK(check_4_local_map->contains(4));
    CHECK(check_4_local_map->find(1) != check_4_local_map->end());
    CHECK(check_4_local_map->find(2) == check_4_local_map->end());
    CHECK(check_4_local_map->find(3) != check_4_local_map->end());
    CHECK(check_4_local_map->find(4) != check_4_local_map->end());
    CHECK(check_4_local_map->find(5) != check_4_local_map->end());
    CHECK(check_4_local_map->find(4) != check_4_local_map->find(5));
    CHECK(check_4_local_map->find(4) == l_it_bool_to_4.first);
    CHECK(check_4_local_map->at(4) == expected);
    CHECK(cpp17::as_const(*check_4_local_map).at(4) == expected);
    make_overloader(
        [expected](gsl::not_null<CopyableKeyMapType*> more_local_map) noexcept {
          CHECK((*more_local_map)[4] == expected);
        },
        [](gsl::not_null<NonCopyableKeyMapType*> /*local_map*/) noexcept {})(
        check_4_local_map);
  };

  make_overloader(
      [&check_4](gsl::not_null<CopyableKeyMapType*> local_map) noexcept {
        // Test serialization and that FixedHashMap works correctly after.
        test_serialization(*local_map);

        auto after_map = serialize_and_deserialize(*local_map);
        CHECK(after_map.size() == 3);
        CHECK_FALSE(after_map.empty());
        CHECK(after_map.count(1) == 1);
        CHECK(after_map.count(2) == 0);
        CHECK(after_map.count(3) == 1);
        CHECK(after_map.count(5) == 1);
        CHECK(after_map.contains(5));
        CHECK(after_map.find(1) != after_map.end());
        CHECK(after_map.find(2) == after_map.end());
        CHECK(after_map.find(3) != after_map.end());
        CHECK(after_map.find(4) == after_map.end());
        CHECK(after_map.find(5) != after_map.end());
        CHECK(after_map.find(4) != after_map.find(5));
        CHECK(after_map.at(5) == 15);
        CHECK(cpp17::as_const(after_map).at(5) == 15);
        CHECK(after_map[5] == 15);

        check_4(after_map.insert_or_assign(4, 14), 14, true, &after_map);
        check_4(after_map.insert_or_assign(4, 24), 24, false, &after_map);

        after_map.erase(4);
        const size_t key_4 = 4;
        check_4(after_map.insert_or_assign(key_4, 34), 34, true, &after_map);
        check_4(after_map.insert_or_assign(key_4, 44), 44, false, &after_map);
      },
      [&check_4](gsl::not_null<NonCopyableKeyMapType*> local_map) noexcept {
        check_4(local_map->insert_or_assign(4, 14), 14, true, local_map);
        check_4(local_map->insert_or_assign(4, 24), 24, false, local_map);

        local_map->erase(4);
        const size_t key_4 = 4;
        check_4(local_map->insert_or_assign(key_4, 34), 34, true, local_map);
        check_4(local_map->insert_or_assign(key_4, 44), 44, false, local_map);
      })(make_not_null(&map));

  test_iterators(map);
  make_overloader(
      [](gsl::not_null<CopyableKeyMapType*> local_map) noexcept {
        test_copy_semantics(*local_map);
        const auto map2 = *local_map;
        test_move_semantics(std::move(*local_map), map2);
      },
      [](gsl::not_null<NonCopyableKeyMapType*> /*local_map*/) noexcept {})(
      make_not_null(&map));
}

SPECTRE_TEST_CASE("Unit.DataStructures.FixedHashMap",
                  "[DataStructures][Unit]") {
  test_direction_key<4>();
  test_repeated_key<size_t>();
  test_repeated_key<NonCopyableSizeT>();
}
}  // namespace
