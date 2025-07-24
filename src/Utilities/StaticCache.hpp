// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsInteger.hpp"

/// \ingroup UtilitiesGroup
/// Range of integral values for StaticCache indices.  The `Start` is inclusive
/// and the `End` is exclusive.  The range must not be empty.
template <auto Start, auto End>
struct CacheRange {
  static_assert(std::is_same_v<decltype(Start), decltype(End)>);
  static_assert(Start < End, "CacheRange must include at least one value");
  constexpr static auto start = Start;
  constexpr static auto end = End;
  constexpr static auto size = end - start;
  using value_type = std::remove_cv_t<decltype(start)>;
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
  static constexpr std::array<value_type, size> values{Enums...};
  using value_list = tmpl::integral_list<EnumerationType, Enums...>;
};

/// \ingroup UtilitiesGroup
/// A cache of objects intended to be stored in a static variable.
///
/// Objects can be accessed via a combination of several `size_t` and `enum`
/// arguments. The range of each integral argument is specified via a template
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
  explicit StaticCache(Gen&& generator)
      : generator_{std::forward<Gen>(generator)} {}

  template <typename... Args>
  const T& operator()(const Args... parameters) const {
    static_assert(sizeof...(parameters) == sizeof...(Ranges),
                  "Number of arguments must match number of ranges.");
    return unwrap_cache_combined(generate_tuple<Ranges>(parameters)...);
  }

 private:
  template <typename Range, typename T1>
  auto generate_tuple(const T1 parameter) const {
    if constexpr (std::is_enum<T1>::value) {
      static_assert(
          std::is_same<typename Range::value_type, std::remove_cv_t<T1>>::value,
          "Mismatched enum parameter type and cached type.");
      size_t array_location = std::numeric_limits<size_t>::max();
      static const std::array<typename Range::value_type, Range::size> values{
          Range::values};
      for (size_t i = 0; i < Range::size; ++i) {
        if (parameter == gsl::at(values, i)) {
          array_location = i;
          break;
        }
      }
      if (UNLIKELY(array_location == std::numeric_limits<size_t>::max())) {
        ERROR("Uncached enumeration value: " << parameter);
      }
      return std::tuple{array_location, typename Range::value_list{}};
    } else {
      static_assert(
          tt::is_integer_v<std::remove_cv_t<T1>>,
          "The parameter passed for a CacheRange must be an integer type.");

      // Check range here because the nested range checks in the unwrap_cache
      // function cause significant compile time overhead.
      if (UNLIKELY(Range::start >
                       static_cast<decltype(Range::start)>(parameter) or
                   static_cast<decltype(Range::start)>(parameter) >=
                       Range::start +
                           static_cast<decltype(Range::start)>(Range::size))) {
        ERROR("Index out of range: "
              << Range::start << " <= " << parameter << " < "
              << Range::start +
                     static_cast<decltype(Range::start)>(Range::size));
      }
      return std::tuple{
          // unsigned cast is safe since this is an index into an array
          static_cast<size_t>(
              static_cast<typename Range::value_type>(parameter) -
              Range::start),
          tmpl::make_sequence<
              tmpl::integral_constant<typename Range::value_type, Range::start>,
              Range::size>{}};
    }
  }

  // Compilation time notes:
  //
  // - The separate peeling of different number of arguments is the
  //   fastest implementation Nils Deppe has found so far.
  // - The second fastest is using a Cartesian product on the lists of
  //   possible values, followed by a for_each over that list to set the
  //   function pointers in the array. Note that having the for_each function
  //   be marked `constexpr` resulted in a 6x reduction in compilation time
  //   for clang 17 compared to the not constexpr version, but still 40%
  //   slower compilation compared to the pattern matching below.
  template <typename... IntegralConstantValues>
  const T& unwrap_cache_combined() const {
    static const T cached_object = generator_(IntegralConstantValues::value...);
    return cached_object;
  }

  // NVCC v12.6 can't handle the unwrapping code below, so we use a simpler
  // version for CUDA. Note that we don't actually need to use StaticCache on
  // device at all, so the main issue with NVCC is that it tries to compile this
  // code for CUDA at all.
#if defined(__NVCC__) && defined(__CUDA_ARCH__)
  template <typename... IntegralConstantValues, typename... IntegralConstants,
            typename... Args>
  const T& unwrap_cache_combined(
      std::tuple<size_t, tmpl::list<IntegralConstants...>> parameter0,
      Args... parameters) const {
    // note that the act of assigning to the specified function pointer type
    // fixes the template arguments that need to be inferred.
    static const std::array<
        const T& (StaticCache<Generator, T, Ranges...>::*)(Args...) const,
        sizeof...(IntegralConstants)>
        cache{{&StaticCache<Generator, T, Ranges...>::unwrap_cache_combined<
            IntegralConstantValues..., IntegralConstants>...}};
    // The array `cache` holds pointers to member functions, so we dereference
    // the pointer and invoke it on `this`.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ > 10 && __GNUC__ < 14
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
    return (this->*gsl::at(cache, std::get<0>(parameter0)))(parameters...);
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ > 10 && __GNUC__ < 14
#pragma GCC diagnostic pop
#endif
  }
#else  // defined(__NVCC__) && defined(__CUDA_ARCH__)
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ > 10 && __GNUC__ < 14
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
  template <typename... IntegralConstantValues, typename... IntegralConstants>
  const T& unwrap_cache_combined(
      std::tuple<size_t, tmpl::list<IntegralConstants...>> parameter0) const {
    // note that the act of assigning to the specified function pointer type
    // fixes the template arguments that need to be inferred.
    static const std::array<const T& (StaticCache::*)() const,
                            sizeof...(IntegralConstants)>
        cache{{&StaticCache::unwrap_cache_combined<IntegralConstantValues...,
                                                   IntegralConstants>...}};
    // The array `cache` holds pointers to member functions, so we dereference
    // the pointer and invoke it on `this`.
    return (this->*gsl::at(cache, std::get<0>(parameter0)))();
  }

  template <typename... IntegralConstantValues, typename... IntegralConstants0,
            typename... IntegralConstants1>
  const T& unwrap_cache_combined(
      std::tuple<size_t, tmpl::list<IntegralConstants0...>> parameter0,
      std::tuple<size_t, tmpl::list<IntegralConstants1...>> parameter1) const {
    constexpr size_t num0 = sizeof...(IntegralConstants0);
    constexpr size_t num1 = sizeof...(IntegralConstants1);
    constexpr size_t total_size = num0 * num1;
    // note that the act of assigning to the specified function pointer type
    // fixes the template arguments that need to be inferred.
    static const std::array<const T& (StaticCache::*)() const, total_size>
        cache = []() {
          std::array<const T& (StaticCache::*)() const, total_size> result;
          size_t counter1 = 0;
          const auto helper1 = [&counter1,
                                &result]<typename IntegralConstant1>() {
            size_t counter0 = 0;
            const auto helper0 = [&counter0, &counter1,
                                  &result]<typename IntegralConstant0>() {
              result[counter0 + num0 * counter1] =
                  &StaticCache::unwrap_cache_combined<IntegralConstantValues...,
                                                      IntegralConstant0,
                                                      IntegralConstant1>;
              ++counter0;
            };
            EXPAND_PACK_LEFT_TO_RIGHT(
                helper0.template operator()<IntegralConstants0>());
            ++counter1;
          };
          EXPAND_PACK_LEFT_TO_RIGHT(
              helper1.template operator()<IntegralConstants1>());
          return result;
        }();

    // The array `cache` holds pointers to member functions, so we dereference
    // the pointer and invoke it on `this`.
    return (this->*gsl::at(cache, std::get<0>(parameter0) +
                                      num0 * std::get<0>(parameter1)))();
  }

  template <typename... IntegralConstantValues, typename... IntegralConstants0,
            typename... IntegralConstants1, typename... IntegralConstants2>
  const T& unwrap_cache_combined(
      std::tuple<size_t, tmpl::list<IntegralConstants0...>> parameter0,
      std::tuple<size_t, tmpl::list<IntegralConstants1...>> parameter1,
      std::tuple<size_t, tmpl::list<IntegralConstants2...>> parameter2) const {
    constexpr size_t num0 = sizeof...(IntegralConstants0);
    constexpr size_t num1 = sizeof...(IntegralConstants1);
    constexpr size_t num2 = sizeof...(IntegralConstants2);
    constexpr size_t total_size = num0 * num1 * num2;
    // note that the act of assigning to the specified function pointer type
    // fixes the template arguments that need to be inferred.
    static const std::array<const T& (StaticCache::*)() const, total_size>
        cache = []() {
          std::array<const T& (StaticCache::*)() const, total_size> result;
          size_t counter2 = 0;
          const auto helper2 = [&counter2,
                                &result]<typename IntegralConstant2>() {
            size_t counter1 = 0;
            const auto helper1 = [&counter1, &counter2,
                                  &result]<typename IntegralConstant1>() {
              size_t counter0 = 0;
              const auto helper0 = [&counter0, &counter1, &counter2,
                                    &result]<typename IntegralConstant0>() {
                result[counter0 + num0 * (counter1 + num1 * counter2)] =
                    &StaticCache::unwrap_cache_combined<
                        IntegralConstantValues..., IntegralConstant0,
                        IntegralConstant1, IntegralConstant2>;
                ++counter0;
              };
              EXPAND_PACK_LEFT_TO_RIGHT(
                  helper0.template operator()<IntegralConstants0>());
              ++counter1;
            };
            EXPAND_PACK_LEFT_TO_RIGHT(
                helper1.template operator()<IntegralConstants1>());
            ++counter2;
          };
          EXPAND_PACK_LEFT_TO_RIGHT(
              helper2.template operator()<IntegralConstants2>());
          return result;
        }();

    // The array `cache` holds pointers to member functions, so we dereference
    // the pointer and invoke it on `this`.
    return (
        this->*gsl::at(cache, std::get<0>(parameter0) +
                                  num0 * (std::get<0>(parameter1) +
                                          num1 * std::get<0>(parameter2))))();
  }

  template <typename... IntegralConstantValues, typename... IntegralConstants0,
            typename... IntegralConstants1, typename... IntegralConstants2,
            typename... IntegralConstants3>
  const T& unwrap_cache_combined(
      std::tuple<size_t, tmpl::list<IntegralConstants0...>> parameter0,
      std::tuple<size_t, tmpl::list<IntegralConstants1...>> parameter1,
      std::tuple<size_t, tmpl::list<IntegralConstants2...>> parameter2,
      std::tuple<size_t, tmpl::list<IntegralConstants3...>> parameter3) const {
    constexpr size_t num0 = sizeof...(IntegralConstants0);
    constexpr size_t num1 = sizeof...(IntegralConstants1);
    constexpr size_t num2 = sizeof...(IntegralConstants2);
    constexpr size_t num3 = sizeof...(IntegralConstants3);
    constexpr size_t total_size = num0 * num1 * num2 * num3;
    // note that the act of assigning to the specified function pointer type
    // fixes the template arguments that need to be inferred.
    static const std::array<const T& (StaticCache::*)() const, total_size>
        cache = []() {
          std::array<const T& (StaticCache::*)() const, total_size> result;
          size_t counter3 = 0;
          const auto helper3 = [&counter3,
                                &result]<typename IntegralConstant3>() {
            size_t counter2 = 0;
            const auto helper2 = [&counter2, &counter3,
                                  &result]<typename IntegralConstant2>() {
              size_t counter1 = 0;
              const auto helper1 = [&counter1, &counter2, &counter3,
                                    &result]<typename IntegralConstant1>() {
                size_t counter0 = 0;
                const auto helper0 = [&counter0, &counter1, &counter2,
                                      &counter3,
                                      &result]<typename IntegralConstant0>() {
                  result[counter0 +
                         num0 *
                             (counter1 + num1 * (counter2 + num2 * counter3))] =
                      &StaticCache::unwrap_cache_combined<
                          IntegralConstantValues..., IntegralConstant0,
                          IntegralConstant1, IntegralConstant2,
                          IntegralConstant3>;
                  ++counter0;
                };
                EXPAND_PACK_LEFT_TO_RIGHT(
                    helper0.template operator()<IntegralConstants0>());
                ++counter1;
              };
              EXPAND_PACK_LEFT_TO_RIGHT(
                  helper1.template operator()<IntegralConstants1>());
              ++counter2;
            };
            EXPAND_PACK_LEFT_TO_RIGHT(
                helper2.template operator()<IntegralConstants2>());
            ++counter3;
          };
          EXPAND_PACK_LEFT_TO_RIGHT(
              helper3.template operator()<IntegralConstants3>());
          return result;
        }();

    // The array `cache` holds pointers to member functions, so we
    // dereference the pointer and invoke it on `this`.
    return (
        this->*gsl::at(cache,
                       std::get<0>(parameter0) +
                           num0 * (std::get<0>(parameter1) +
                                   num1 * (std::get<0>(parameter2) +
                                           num2 * std::get<0>(parameter3)))))();
  }

  template <typename... IntegralConstantValues, typename... IntegralConstants0,
            typename... IntegralConstants1, typename... IntegralConstants2,
            typename... IntegralConstants3, typename... IntegralConstants4,
            typename... Args>
  const T& unwrap_cache_combined(
      std::tuple<size_t, tmpl::list<IntegralConstants0...>> parameter0,
      std::tuple<size_t, tmpl::list<IntegralConstants1...>> parameter1,
      std::tuple<size_t, tmpl::list<IntegralConstants2...>> parameter2,
      std::tuple<size_t, tmpl::list<IntegralConstants3...>> parameter3,
      std::tuple<size_t, tmpl::list<IntegralConstants4...>> parameter4,
      const Args&... parameters) const {
    constexpr size_t num0 = sizeof...(IntegralConstants0);
    constexpr size_t num1 = sizeof...(IntegralConstants1);
    constexpr size_t num2 = sizeof...(IntegralConstants2);
    constexpr size_t num3 = sizeof...(IntegralConstants3);
    constexpr size_t num4 = sizeof...(IntegralConstants4);
    constexpr size_t total_size = num0 * num1 * num2 * num3 * num4;
    // note that the act of assigning to the specified function pointer type
    // fixes the template arguments that need to be inferred.
    static const std::array<const T& (StaticCache::*)(Args...) const,
                            total_size>
        cache = []() {
          std::array<const T& (StaticCache::*)(Args...) const, total_size>
              result;
          size_t counter4 = 0;
          const auto helper4 = [&counter4,
                                &result]<typename IntegralConstant4>() {
            size_t counter3 = 0;
            const auto helper3 = [&counter3, &counter4,
                                  &result]<typename IntegralConstant3>() {
              size_t counter2 = 0;
              const auto helper2 = [&counter2, &counter3, &counter4,
                                    &result]<typename IntegralConstant2>() {
                size_t counter1 = 0;
                const auto helper1 = [&counter1, &counter2, &counter3,
                                      &counter4,
                                      &result]<typename IntegralConstant1>() {
                  size_t counter0 = 0;
                  const auto helper0 = [&counter0, &counter1, &counter2,
                                        &counter3, &counter4,
                                        &result]<typename IntegralConstant0>() {
                    result[counter0 +
                           num0 *
                               (counter1 +
                                num1 * (counter2 +
                                        num2 * (counter3 + num3 * counter4)))] =
                        &StaticCache::unwrap_cache_combined<
                            IntegralConstantValues..., IntegralConstant0,
                            IntegralConstant1, IntegralConstant2,
                            IntegralConstant3, IntegralConstant4>;
                    ++counter0;
                  };
                  EXPAND_PACK_LEFT_TO_RIGHT(
                      helper0.template operator()<IntegralConstants0>());
                  ++counter1;
                };
                EXPAND_PACK_LEFT_TO_RIGHT(
                    helper1.template operator()<IntegralConstants1>());
                ++counter2;
              };
              EXPAND_PACK_LEFT_TO_RIGHT(
                  helper2.template operator()<IntegralConstants2>());
              ++counter3;
            };
            EXPAND_PACK_LEFT_TO_RIGHT(
                helper3.template operator()<IntegralConstants3>());
            ++counter4;
          };
          EXPAND_PACK_LEFT_TO_RIGHT(
              helper4.template operator()<IntegralConstants4>());
          return result;
        }();

    // The array `cache` holds pointers to member functions, so we dereference
    // the pointer and invoke it on `this`.
    return (
        this->*gsl::at(
                   cache,
                   std::get<0>(parameter0) +
                       num0 *
                           (std::get<0>(parameter1) +
                            num1 * (std::get<0>(parameter2) +
                                    num2 * (std::get<0>(parameter3) +
                                            num3 * std::get<0>(parameter4))))))(
        parameters...);
  }
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ > 10 && __GNUC__ < 14
#pragma GCC diagnostic pop
#endif
#endif  // defined(__NVCC__) && defined(__CUDA_ARCH__)

  const Generator generator_;
};

/// \ingroup UtilitiesGroup
/// Create a StaticCache, inferring the cached type from the generator.
template <typename... Ranges, typename Generator>
auto make_static_cache(Generator&& generator) {
  using CachedType = std::remove_cv_t<decltype(generator(
      std::declval<typename Ranges::value_type>()...))>;
  return StaticCache<std::remove_cv_t<Generator>, CachedType, Ranges...>(
      std::forward<Generator>(generator));
}
