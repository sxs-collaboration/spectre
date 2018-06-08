// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

// IWYU pragma: no_include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/StdHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
template <size_t Dim>
void check_single_element_map(const Direction<Dim>& direction) noexcept {
  DirectionMap<Dim, std::pair<double, NonCopyable>> map;
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
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.DirectionMap", "[Domain][Unit]") {
  for (const auto& direction : Direction<2>::all_directions()) {
    check_single_element_map(direction);
  }

  DirectionMap<2, double> map;

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
      CHECK(e.what() == get_output(dir3) + " not in map");
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
    DirectionMap<2, NonCopyable> nc_map;
    nc_map.emplace(Direction<2>::upper_xi(), NonCopyable{});
    DirectionMap<2, NonCopyable> nc_map2;
    nc_map2.emplace(Direction<2>::upper_xi(), NonCopyable{});

    check_equal(nc_map, nc_map2);
    auto nc_map3 = std::move(nc_map2);
    check_equal(nc_map, nc_map3);
    DirectionMap<2, NonCopyable> nc_map4;
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
    CHECK(decltype(map)(std::initializer_list<decltype(map)::value_type>{})
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

  CHECK(map.erase(dir1) == 1);
  CHECK(map.size() == 2);
  CHECK(map.find(dir1) == map.end());
  CHECK(map.erase(dir1) == 0);
  CHECK(map.size() == 2);
  CHECK(map.find(dir1) == map.end());

  map.clear();
  CHECK(map.empty());
  CHECK(map.size() == 0);  // NOLINT(readability-container-size-empty)
  test_iterators(map);
}
