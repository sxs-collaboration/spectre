// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <thread>
#include <utility>
#include <vector>

#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/RuntimeCache.hpp"

namespace {
enum class Color { Red, Green, Purple };

std::ostream& operator<<(std::ostream& os, Color t) {
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

std::ostream& operator<<(std::ostream& os, Animal t) {
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
  // [static_cache]
  const static auto cache =
      make_static_cache<CacheRange<0_st, 3_st>, CacheRange<3_st, 5_st>>(
          [](const size_t a, const size_t b) { return a + b; });
  CHECK(cache(0, 3) == 3);  // smallest entry
  CHECK(cache(2, 4) == 6);  // largest entry
  // [static_cache]

  std::vector<std::pair<size_t, size_t>> calls;
  const auto cache2 =
      make_static_cache<CacheRange<0_st, 3_st>, CacheRange<3_st, 5_st>>(
          [&calls](const size_t a, const size_t b) {
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
  const auto small_cache = make_static_cache([&small_calls]() {
    ++small_calls;
    return size_t{5};
  });
  CHECK(small_calls == 0);
  CHECK(small_cache() == 5);
  CHECK(small_calls == 1);

  // [static_cache_no_args]
  const auto simple_small_cache =
      make_static_cache([]() { return size_t{10}; });
  CHECK(simple_small_cache() == 10);
  // [static_cache_no_args]

  // check enum caching functionality
  const auto enum_generator_tuple = [](const Color& color,
                                       const size_t value = 5,
                                       const Animal animal =
                                           Animal::Goldendoodle) {
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
  // [static_cache_with_enum]
  const auto simple_enum_cache = make_static_cache<
      CacheEnumeration<Color, Color::Red, Color::Green, Color::Purple>>(
      [](const Color color) { return std::string{MakeString{} << color}; });
  CHECK(simple_enum_cache(Color::Red) == "Red");
  // [static_cache_with_enum]

  const auto int_cache = make_static_cache<CacheRange<-5, 10>>(
      [](const int val) { return pow<3>(val); });
  CHECK(int_cache(-3) == -27);
  CHECK(int_cache(2) == 8);

  const auto enum_cache = make_static_cache<
      CacheEnumeration<Color, Color::Red, Color::Green, Color::Purple>>(
      enum_generator_tuple);
  for (const auto color : {Color::Red, Color::Green, Color::Purple}) {
    CHECK(enum_cache(color) ==
          std::make_tuple(3, static_cast<size_t>(color) + 1, 5));
  }

  const auto enum_size_t_cache = make_static_cache<
      CacheEnumeration<Color, Color::Red, Color::Green, Color::Purple>,
      CacheRange<3_st, 5_st>>(enum_generator_tuple);
  for (const auto color : {Color::Red, Color::Green, Color::Purple}) {
    CHECK(enum_size_t_cache(color, 3) ==
          std::make_tuple(3, static_cast<size_t>(color) + 1, 3));
    CHECK(enum_size_t_cache(color, 4) ==
          std::make_tuple(3, static_cast<size_t>(color) + 1, 4));
  }

  // [static_cache_with_enum_and_numeric]
  const auto simple_enum_size_t_enum_cache = make_static_cache<
      CacheEnumeration<Color, Color::Red, Color::Green, Color::Purple>,
      CacheRange<3_st, 5_st>,
      CacheEnumeration<Animal, Animal::Goldendoodle, Animal::Labradoodle,
                       Animal::Poodle>>(
      [](const Color color, const size_t value, const Animal animal) {
        return std::string{MakeString{} << color << value << animal};
      });
  CHECK(simple_enum_size_t_enum_cache(Color::Red, 3, Animal::Labradoodle) ==
        "Red3Labradoodle");
  CHECK(simple_enum_size_t_enum_cache(Color::Purple, 4, Animal::Poodle) ==
        "Purple4Poodle");
  // [static_cache_with_enum_and_numeric]
  const auto enum_size_t_enum_cache = make_static_cache<
      CacheEnumeration<Color, Color::Red, Color::Green, Color::Purple>,
      CacheRange<3_st, 5_st>,
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

  // RuntimeCache has the same lookup semantics as StaticCache, but avoids
  // instantiating the generator for every combination of indices.
  std::vector<std::pair<size_t, Color>> runtime_calls;
  const auto runtime_cache = make_runtime_cache<
      CacheRange<2_st, 5_st>,
      CacheEnumeration<Color, Color::Red, Color::Green, Color::Purple>>(
      [&runtime_calls](const size_t value, const Color color) {
        runtime_calls.emplace_back(value, color);
        return std::string{MakeString{} << value << color};
      });
  CHECK(runtime_calls.empty());
  CHECK(runtime_cache(2, Color::Red) == "2Red");
  CHECK(runtime_cache(4, Color::Purple) == "4Purple");
  CHECK(runtime_calls.size() == 2);
  CHECK(runtime_cache(2, Color::Red) == "2Red");
  CHECK(runtime_calls.size() == 2);

  size_t runtime_small_calls = 0;
  const auto runtime_small_cache = make_runtime_cache([&runtime_small_calls]() {
    ++runtime_small_calls;
    return size_t{10};
  });
  CHECK(runtime_small_calls == 0);
  CHECK(runtime_small_cache() == 10);
  CHECK(runtime_small_cache() == 10);
  CHECK(runtime_small_calls == 1);

  std::atomic<size_t> concurrent_calls{0};
  const auto concurrent_cache = make_runtime_cache<CacheRange<0_st, 1_st>>(
      [&concurrent_calls](const size_t value) {
        ++concurrent_calls;
        return value + 10;
      });
  std::atomic<bool> start_concurrent_calls{false};
  std::array<size_t, 8> concurrent_results{};
  std::array<std::thread, 8> threads;
  for (size_t i = 0; i < threads.size(); ++i) {
    gsl::at(threads, i) = std::thread{
        [&concurrent_cache, &concurrent_results, &start_concurrent_calls, i]() {
          while (not start_concurrent_calls.load(std::memory_order_acquire)) {
          }
          gsl::at(concurrent_results, i) = concurrent_cache(0);
        }};
  }
  start_concurrent_calls.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }
  CHECK(std::all_of(concurrent_results.begin(), concurrent_results.end(),
                    [](const size_t result) { return result == 10; }));
  CHECK(concurrent_calls == 1);

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      (make_static_cache<CacheRange<3, 5>>([](const size_t x) { return x; })(
          2)),
      Catch::Matchers::ContainsSubstring("Index out of range: 3 <= 2 < 5"));
  CHECK_THROWS_WITH(
      (make_static_cache<CacheRange<3, 5>>([](const size_t x) { return x; })(
          5)),
      Catch::Matchers::ContainsSubstring("Index out of range: 3 <= 5 < 5"));
  CHECK_THROWS_WITH(
      (make_runtime_cache<CacheRange<3, 5>>([](const int x) { return x; })(
          2)),
      Catch::Matchers::ContainsSubstring("Index out of range: 3 <= 2 < 5"));
  CHECK_THROWS_WITH(
      (make_runtime_cache<CacheEnumeration<Color, Color::Red, Color::Green>>(
          [](const Color color) { return color; })(Color::Purple)),
      Catch::Matchers::ContainsSubstring("Uncached enumeration value: Purple"));
#endif

  // Test that the passed callable is stored, so we don't get a
  // dangling reference in the usual case that the cache outlives its
  // calling scope.
  {
    struct StoreTest {
      int value{};
      int operator()() const { return value; }
    };

    StoreTest callable{5};
    const auto cache_from_lvalue = make_static_cache(callable);
    callable.value = 8;
    CHECK(cache_from_lvalue() == 5);
  }

  // Test that the cache caches values, even if the callable returns
  // by reference
  {
    int value = 5;
    const auto cache_returned_ref =
        make_static_cache([&value]() -> const int& { return value; });
    CHECK(cache_returned_ref() == 5);
    value = 8;
    CHECK(cache_returned_ref() == 5);
  }
}
