// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"

namespace StaticCache_detail {
template <typename T, typename... Ranges>
class StaticCacheImpl;

template <typename T, typename Range0, typename... Ranges>
class StaticCacheImpl<T, Range0, Ranges...> {
  template <typename>
  struct construction_helper;

  template <size_t... RangeValues>
  struct construction_helper<std::index_sequence<RangeValues...>> {
    template <typename Generator, typename... Args>
    constexpr static std::array<StaticCacheImpl<T, Ranges...>, Range0::size>
    construct(Generator&& generator, const Args... args) noexcept {
      return {{StaticCacheImpl<T, Ranges...>(generator, args...,
                                             RangeValues + Range0::start)...}};
    }
  };

 public:
  template <typename Generator, typename... Args,
            Requires<not cpp17::is_same_v<std::decay_t<Generator>,
                                          StaticCacheImpl>> = nullptr>
  explicit constexpr StaticCacheImpl(Generator&& generator,
                                     const Args... args) noexcept
      : data_(construction_helper<
              std::make_index_sequence<Range0::size>>::construct(generator,
                                                                 args...)) {}

  template <typename... Args>
  const T& operator()(const size_t first_index, const Args... rest) const
      noexcept {
    ASSERT(Range0::start <= first_index and first_index < Range0::end,
           "Index out of range: " << Range0::start << " <= " << first_index
           << " < " << Range0::end);
    return gsl::at(data_, first_index - Range0::start)(rest...);
  }

 private:
  std::array<StaticCacheImpl<T, Ranges...>, Range0::size> data_;
};

template <typename T>
class StaticCacheImpl<T> {
 public:
  template <typename Generator, typename... Args,
            Requires<not cpp17::is_same_v<std::decay_t<Generator>,
                                          StaticCacheImpl>> = nullptr>
  explicit constexpr StaticCacheImpl(Generator&& generator,
                                     const Args... args) noexcept
      : data_(generator(args...)) {}

  const T& operator()() const noexcept { return data_; }

 private:
  T data_;
};
}  // namespace StaticCache_detail

/// \ingroup UtilitiesGroup
/// A cache of objects intended to be stored in a static variable.
///
/// Objects can be accessed using several `size_t` arguments in the
/// ranges specified as template parameters.  The CacheRange template
/// parameters give first and one-past-last values for the valid
/// ranges for each argument.
///
/// \example
/// \snippet Test_StaticCache.cpp static_cache
///
/// \see make_static_cache
///
/// \tparam T type held in the cache
/// \tparam Ranges ranges of valid indices
template <typename T, typename... Ranges>
class StaticCache {
 public:
  /// Initialize the cache.  All objects will be created by calling
  /// `generator` before the constructor returns.
  // clang-tidy: misc-forwarding-reference-overload - fixed with
  // Requires, but clang-tidy can't recognize that.
  template <typename Generator,
            Requires<not cpp17::is_same_v<std::decay_t<Generator>,
                                          StaticCache>> = nullptr>
  explicit constexpr StaticCache(Generator&& generator) noexcept  // NOLINT
      : data_(generator) {}

  template <typename... Args>
  const T& operator()(const Args... indices) const noexcept {
    static_assert(sizeof...(Args) == sizeof...(Ranges),
                  "Number of arguments must match number of ranges.");
    return data_(static_cast<size_t>(indices)...);
  }

 private:
  StaticCache_detail::StaticCacheImpl<T, Ranges...> data_;
};

/// \ingroup UtilitiesGroup
/// Create a StaticCache, inferring the cached type from the generator.
template <typename... Ranges, typename Generator>
auto make_static_cache(Generator&& generator) noexcept {
  using CachedType = std::decay_t<decltype(generator((Ranges{}, size_t{})...))>;
  return StaticCache<CachedType, Ranges...>(generator);
}

/// \ingroup UtilitiesGroup
/// Range of values for StaticCache indices.  The `Start` is inclusive
/// and the `End` is exclusive.  The range must not be empty.
template <size_t Start, size_t End>
struct CacheRange {
  static_assert(Start < End, "CacheRange must include at least one value");
  constexpr static size_t start = Start;
  constexpr static size_t end = End;
  constexpr static size_t size = end - start;
};
