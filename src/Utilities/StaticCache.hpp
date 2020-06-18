// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits/IsInteger.hpp"

/// \ingroup UtilitiesGroup
/// Range of values for StaticCache indices.  The `Start` is inclusive
/// and the `End` is exclusive.  The range must not be empty.
template <size_t Start, size_t End>
struct CacheRange {
  static_assert(Start < End, "CacheRange must include at least one value");
  constexpr static size_t start = Start;
  constexpr static size_t end = End;
  constexpr static size_t size = end - start;
  using value_type = size_t;
};

/// \ingroup UtilitiesGroup
/// Possible enumeration values for the StaticCache. Only values specified here
/// are retrievable.
///
/// \note The `EnumerationType` must be streamable.
template <typename EnumerationType, EnumerationType... Enums>
struct CacheEnumeration {
  constexpr static size_t size = sizeof...(Enums);
  using value_type = EnumerationType;
};

/// \ingroup UtilitiesGroup
/// A cache of objects intended to be stored in a static variable.
///
/// Objects can be accessed via a combination of several `size_t` and `enum`
/// arguments. The range of each `size_t` argument is specified via a template
/// parameter of type `CacheRange<start, end>`, giving the first and
/// one-past-last values for the range. Each `enum` argument is specified by a
/// template parameter of type `CacheEnumeration<EnumerationType, Members...>`
/// giving the enumeration type and an explicit set of every enum member to be
/// cached.
///
/// \example
/// A cache with only numeric indices:
/// \snippet Test_StaticCache.cpp static_cache
///
/// \example
/// A cache with enumeration indices:
/// \snippet Test_StaticCache.cpp static_cache_with_enum
///
/// \example
/// A cache with mixed numeric and enumeration indices:
/// \snippet Test_StaticCache.cpp static_cache_with_enum_and_numeric
///
/// \example
/// A cache with no arguments at all (caching only a single object)
/// \snippet Test_StaticCache.cpp static_cache_no_args
///
/// \see make_static_cache
///
/// \tparam T type held in the cache
/// \tparam Ranges ranges of valid indices
template <typename Generator, typename T, typename... Ranges>
class StaticCache {
 public:
  template <typename Gen>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  explicit StaticCache(Gen&& generator) noexcept
      : generator_{std::forward<Gen>(generator)} {}

  template <typename... Args>
  const T& operator()(const Args... parameters) const noexcept {
    static_assert(sizeof...(parameters) == sizeof...(Ranges),
                  "Number of arguments must match number of ranges.");
    return unwrap_cache(generate_tuple<Ranges>(parameters)...);
  }

 private:
  template <typename Range, typename T1,
            Requires<not std::is_enum<T1>::value> = nullptr>
  auto generate_tuple(const T1 parameter) const noexcept {
    static_assert(
        tt::is_integer_v<std::decay_t<T1>>,
        "The parameter passed for a CacheRange must be an integer type.");
    return std::make_tuple(static_cast<size_t>(parameter),
                           std::integral_constant<size_t, Range::start>{},
                           std::make_index_sequence<Range::size>{});
  }

  template <typename Range, typename T1,
            Requires<std::is_enum<T1>::value> = nullptr>
  std::tuple<std::decay_t<T1>, Range> generate_tuple(const T1 parameter) const
      noexcept {
    static_assert(
        std::is_same<typename Range::value_type, std::decay_t<T1>>::value,
        "Mismatched enum parameter type and cached type.");
    return {parameter, Range{}};
  }

  template <typename... IntegralConstantValues>
  const T& unwrap_cache() const noexcept {
    static const T cached_object = generator_(IntegralConstantValues::value...);
    return cached_object;
  }

  template <typename... IntegralConstantValues, size_t IndexOffset,
            size_t... Is, typename... Args>
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
            IntegralConstantValues...,
            std::integral_constant<size_t, Is + IndexOffset>>...}};
    // The array `cache` holds pointers to member functions, so we dereference
    // the pointer and invoke it on `this`.
    return (this->*gsl::at(cache, std::get<0>(parameter0) - IndexOffset))(
        parameters...);
  }

  template <typename... IntegralConstantValues, typename EnumType,
            EnumType... EnumValues, typename... Args>
  const T& unwrap_cache(
      std::tuple<EnumType, CacheEnumeration<EnumType, EnumValues...>>
          parameter0,
      Args... parameters) const noexcept {
    size_t array_location = std::numeric_limits<size_t>::max();
    static const std::array<EnumType, sizeof...(EnumValues)> values{
        {EnumValues...}};
    for (size_t i = 0; i < sizeof...(EnumValues); ++i) {
      if (std::get<0>(parameter0) == gsl::at(values, i)) {
        array_location = i;
        break;
      }
    }
    if (UNLIKELY(array_location == std::numeric_limits<size_t>::max())) {
      ERROR("Uncached enumeration value: " << std::get<0>(parameter0));
    }
    // note that the act of assigning to the specified function pointer type
    // fixes the template arguments that need to be inferred.
    static const std::array<
        const T& (StaticCache<Generator, T, Ranges...>::*)(Args...) const,
        sizeof...(EnumValues)>
        cache{{&StaticCache<Generator, T, Ranges...>::unwrap_cache<
            IntegralConstantValues...,
            std::integral_constant<EnumType, EnumValues>>...}};
    // The array `cache` holds pointers to member functions, so we dereference
    // the pointer and invoke it on `this`.
    return (this->*gsl::at(cache, array_location))(parameters...);
  }

  const Generator generator_;
};

/// \ingroup UtilitiesGroup
/// Create a StaticCache, inferring the cached type from the generator.
template <typename... Ranges, typename Generator>
auto make_static_cache(Generator&& generator) noexcept {
  using CachedType = std::decay_t<decltype(
      generator(std::declval<typename Ranges::value_type>()...))>;
  return StaticCache<std::decay_t<Generator>, CachedType, Ranges...>(
      std::forward<Generator>(generator));
}
