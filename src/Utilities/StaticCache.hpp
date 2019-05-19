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
template <typename Generator, typename T, typename... Ranges>
class StaticCache {
 public:
  explicit StaticCache(Generator&& generator) : generator_{generator} {}

  template <typename... Args>
  const T& operator()(const Args... parameters) const noexcept {
    static_assert(sizeof...(parameters) == sizeof...(Ranges),
                  "Number of arguments must match number of ranges.");
    return unwrap_cache(
        std::make_tuple(static_cast<size_t>(parameters),
                        std::integral_constant<size_t, Ranges::start>{},
                        std::make_index_sequence<Ranges::size>{})...);
  }

 private:
  template <size_t... Indices>
  const T& unwrap_cache() const noexcept {
    static const T cached_object = generator_(Indices...);
    return cached_object;
  }

  template <size_t... Indices, size_t IndexOffset, size_t... Is,
            typename... Args>
  const T& unwrap_cache(
      std::tuple<size_t, std::integral_constant<size_t, IndexOffset>,
                 std::index_sequence<Is...>>
          parameter0,
      Args... parameters) const noexcept {
    if (UNLIKELY(IndexOffset > std::get<0>(parameter0) or
                 std::get<0>(parameter0) >= IndexOffset + sizeof...(Is))) {
      ERROR("Index out of range: " << IndexOffset
                                   << " <= " << std::get<0>(parameter0) << " < "
                                   << IndexOffset + sizeof...(Is));
    }
    // note that the act of assigning to the specified function pointer type
    // fixes the template arguments that need to be inferred.
    static const std::array<
        const T& (StaticCache<Generator, T, Ranges...>::*)(Args...) const,
        sizeof...(Is)>
        cache{{&StaticCache<Generator, T, Ranges...>::unwrap_cache<
            Indices..., Is + IndexOffset>...}};
    return (this->*gsl::at(cache, std::get<0>(parameter0) - IndexOffset))(
        parameters...);
  }
  const Generator generator_;
};

/// \ingroup UtilitiesGroup
/// Create a StaticCache, inferring the cached type from the generator.
template <typename... Ranges, typename Generator>
auto make_static_cache(Generator&& generator) noexcept {
  using CachedType = std::decay_t<decltype(generator((Ranges{}, size_t{})...))>;
  return StaticCache<Generator, CachedType, Ranges...>(
      std::forward<Generator>(generator));
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
