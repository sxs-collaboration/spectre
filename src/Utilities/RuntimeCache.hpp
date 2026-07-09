// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StaticCache.hpp"
#include "Utilities/TypeTraits/IsInteger.hpp"

/// \ingroup UtilitiesGroup
/// A cache that selects and lazily initializes entries at runtime.
///
/// Unlike `StaticCache`, this class does not instantiate the generator once for
/// every combination of indices. This reduces compile time for large caches at
/// the cost of runtime index calculation and synchronization.
///
/// \note `RuntimeCache` is tested in `Test_StaticCache.cpp`.
template <typename Generator, typename T, typename... Ranges>
class RuntimeCache {
 public:
  explicit RuntimeCache(Generator generator)
      : generator_{std::move(generator)} {}

  template <typename... Args>
  const T& operator()(const Args... parameters) const {
    static_assert(sizeof...(parameters) == sizeof...(Ranges),
                  "Number of arguments must match number of ranges.");
    size_t array_location = 0;
    ((array_location = array_location * static_cast<size_t>(Ranges::size) +
                       index<Ranges>(parameters)),
     ...);
    std::call_once(gsl::at(initialized_, array_location), [this, array_location,
                                                           &parameters...]() {
      gsl::at(cached_objects_, array_location)
          .emplace(generator_(
              static_cast<typename Ranges::value_type>(parameters)...));
    });
    return gsl::at(cached_objects_, array_location).value();
  }

 private:
  template <typename Range, typename U>
  static size_t index(const U parameter) {
    if constexpr (std::is_enum_v<U>) {
      static_assert(
          std::is_same_v<typename Range::value_type, std::remove_cv_t<U>>,
          "Mismatched enum parameter type and cached type.");
      for (size_t i = 0; i < Range::size; ++i) {
        if (parameter == gsl::at(Range::values, i)) {
          return i;
        }
      }
      ERROR("Uncached enumeration value: " << parameter);
    } else {
      static_assert(
          tt::is_integer_v<std::remove_cv_t<U>>,
          "The parameter passed for a CacheRange must be an integer type.");
      if (UNLIKELY(
              Range::start > static_cast<decltype(Range::start)>(parameter) or
              static_cast<decltype(Range::start)>(parameter) >= Range::end)) {
        ERROR("Index out of range: " << Range::start << " <= " << parameter
                                     << " < " << Range::end);
      }
      return static_cast<size_t>(
          static_cast<typename Range::value_type>(parameter) - Range::start);
    }
  }

  static constexpr size_t number_of_entries =
      (size_t{1} * ... * static_cast<size_t>(Ranges::size));
  Generator generator_;
  // NOLINTNEXTLINE(spectre-mutable)
  mutable std::array<std::once_flag, number_of_entries> initialized_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable std::array<std::optional<T>, number_of_entries> cached_objects_{};
};

/// \ingroup UtilitiesGroup
/// Create a RuntimeCache, inferring the cached type from the generator.
template <typename... Ranges, typename Generator>
auto make_runtime_cache(Generator&& generator) {
  using CachedType = std::remove_cvref_t<decltype(generator(
      std::declval<typename Ranges::value_type>()...))>;
  return RuntimeCache<std::remove_cvref_t<Generator>, CachedType, Ranges...>(
      std::forward<Generator>(generator));
}
