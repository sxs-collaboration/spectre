// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/StaticCache.hpp"

namespace {
enum class Color { Red, Green, Purple };

std::ostream& operator<<(std::ostream& os, Color t) noexcept {
  switch (t) {
    case Color::Red:
      return os << "Red";
    case Color::Green:
      return os << "Green";
    case Color::Purple:
      return os << "Purple";
    default:
      ERROR("Unknown color");
  }
}

enum class Animal { Goldendoodle, Labradoodle, Poodle };

std::ostream& operator<<(std::ostream& os, Animal t) noexcept {
  switch (t) {
    case Animal::Goldendoodle:
      return os << "Goldendoodle";
    case Animal::Labradoodle:
      return os << "Labradoodle";
    case Animal::Poodle:
      return os << "Poodle";
    default:
      ERROR("Unknown Animal");
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.StaticCache", "[Utilities][Unit]") {
  /// [static_cache]
  const static auto cache =
      make_static_cache<CacheRange<0, 3>, CacheRange<3, 5>>(
          [](const size_t a, const size_t b) noexcept { return a + b; });
  CHECK(cache(0, 3) == 3);  // smallest entry
  CHECK(cache(2, 4) == 6);  // largest entry
  /// [static_cache]

  std::vector<std::pair<size_t, size_t>> calls;
  const auto cache2 =
      make_static_cache<CacheRange<0, 3>, CacheRange<3, 5>>([&calls](
          const size_t a, const size_t b) noexcept {
        calls.emplace_back(a, b);
        return a + b;
      });
  // cache is lazy, shouldn't have called at all before retrieving
  CHECK(calls.empty());

  // explicitly call the cache creation to check its contents
  cache2(0, 3);
  cache2(0, 4);
  cache2(1, 3);
  CHECK(calls.size() == 3);

  cache2(0, 4);
  cache2(1, 3);
  cache2(1, 3);
  CHECK(calls.size() == 3);

  cache2(1, 4);
  cache2(2, 3);
  cache2(2, 4);
  CHECK(calls.size() == 6);

  const decltype(calls) expected_calls{{0, 3}, {0, 4}, {1, 3},
                                       {1, 4}, {2, 3}, {2, 4}};
  CHECK(calls == expected_calls);
  for (const auto& call : expected_calls) {
    CHECK(cache2(call.first, call.second) == call.first + call.second);
  }
  CHECK(calls == expected_calls);

  size_t small_calls = 0;
  const auto small_cache = make_static_cache([&small_calls]() noexcept {
    ++small_calls;
    return size_t{5};
  });
  CHECK(small_calls == 0);
  CHECK(small_cache() == 5);
  CHECK(small_calls == 1);

  /// [static_cache_no_args]
  const auto simple_small_cache =
      make_static_cache([]() noexcept { return size_t{10}; });
  CHECK(simple_small_cache() == 10);
  /// [static_cache_no_args]

  // check enum caching functionality
  const auto enum_generator_tuple =
      [](const Color& color, const size_t value = 5,
         const Animal animal = Animal::Goldendoodle) noexcept {
    size_t offset_animal = 0;
    switch (animal) {
      case Animal::Goldendoodle:
        offset_animal = 3;
        break;
      case Animal::Labradoodle:
        offset_animal = 2;
        break;
      case Animal::Poodle:
        offset_animal = 1;
        break;
      default:
        offset_animal = std::numeric_limits<size_t>::max();
        break;
    };

    switch (color) {
      case Color::Red:
        return std::make_tuple(offset_animal, 1, value);
      case Color::Green:
        return std::make_tuple(offset_animal, 2, value);
      case Color::Purple:
        return std::make_tuple(offset_animal, 3, value);
      default:
        return std::make_tuple(offset_animal, std::numeric_limits<int>::max(),
                               value);
    };
  };
  /// [static_cache_with_enum]
  const auto simple_enum_cache = make_static_cache<
      CacheEnumeration<Color, Color::Red, Color::Green, Color::Purple>>([
  ](const Color color) noexcept { return std::string{MakeString{} << color}; });
  CHECK(simple_enum_cache(Color::Red) == "Red");
  /// [static_cache_with_enum]
  const auto enum_cache = make_static_cache<
      CacheEnumeration<Color, Color::Red, Color::Green, Color::Purple>>(
      enum_generator_tuple);
  for (const auto color : {Color::Red, Color::Green, Color::Purple}) {
    CHECK(enum_cache(color) ==
          std::make_tuple(3, static_cast<size_t>(color) + 1, 5));
  }

  const auto enum_size_t_cache = make_static_cache<
      CacheEnumeration<Color, Color::Red, Color::Green, Color::Purple>,
      CacheRange<3, 5>>(enum_generator_tuple);
  for (const auto color : {Color::Red, Color::Green, Color::Purple}) {
    CHECK(enum_size_t_cache(color, 3) ==
          std::make_tuple(3, static_cast<size_t>(color) + 1, 3));
    CHECK(enum_size_t_cache(color, 4) ==
          std::make_tuple(3, static_cast<size_t>(color) + 1, 4));
  }

  /// [static_cache_with_enum_and_numeric]
  const auto simple_enum_size_t_enum_cache = make_static_cache<
      CacheEnumeration<Color, Color::Red, Color::Green, Color::Purple>,
      CacheRange<3, 5>,
      CacheEnumeration<Animal, Animal::Goldendoodle, Animal::Labradoodle,
                       Animal::Poodle>>(
      [](const Color color, const size_t value, const Animal animal) noexcept {
        return std::string{MakeString{} << color << value << animal};
      });
  CHECK(simple_enum_size_t_enum_cache(Color::Red, 3, Animal::Labradoodle) ==
        "Red3Labradoodle");
  CHECK(simple_enum_size_t_enum_cache(Color::Purple, 4, Animal::Poodle) ==
        "Purple4Poodle");
  /// [static_cache_with_enum_and_numeric]
  const auto enum_size_t_enum_cache = make_static_cache<
      CacheEnumeration<Color, Color::Red, Color::Green, Color::Purple>,
      CacheRange<3, 5>,
      CacheEnumeration<Animal, Animal::Goldendoodle, Animal::Labradoodle,
                       Animal::Poodle>>(enum_generator_tuple);
  for (const auto color : {Color::Red, Color::Green, Color::Purple}) {
    for (const auto animal :
         {Animal::Goldendoodle, Animal::Labradoodle, Animal::Poodle}) {
      CHECK(enum_size_t_enum_cache(color, 3, animal) ==
            std::make_tuple(3 - static_cast<size_t>(animal),
                            static_cast<size_t>(color) + 1, 3));
      CHECK(enum_size_t_enum_cache(color, 4, animal) ==
            std::make_tuple(3 - static_cast<size_t>(animal),
                            static_cast<size_t>(color) + 1, 4));
    }
  }
}

// [[OutputRegex, Index out of range: 3 <= 2 < 5]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Utilities.StaticCache.out_of_range.low",
                               "[Utilities][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const auto cache = make_static_cache<CacheRange<3, 5>>(
      [](const size_t x) noexcept { return x; });
  cache(2);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Index out of range: 3 <= 5 < 5]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Utilities.StaticCache.out_of_range.high",
                               "[Utilities][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const auto cache = make_static_cache<CacheRange<3, 5>>(
      [](const size_t x) noexcept { return x; });
  cache(5);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
