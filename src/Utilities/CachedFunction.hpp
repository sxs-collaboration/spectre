// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class CachedFunction

#pragma once

#include <type_traits>
#include <unordered_map>
#include <utility>

/// A function wrapper that caches function values.
template <typename Function, typename Map>
class CachedFunction {
 public:
  using input = typename Map::key_type;
  using output = typename Map::mapped_type;

  template <typename... MapArgs>
  explicit CachedFunction(Function function, MapArgs... map_args) noexcept
      : function_(std::move(function)),
        cache_(std::forward<MapArgs>(map_args)...) {}

  /// Obtain the function result
  const output& operator()(
      const input& x) noexcept(noexcept(std::declval<Function>()(x))) {
    auto it = cache_.find(x);
    if (it == cache_.end()) {
      it = cache_.emplace(x, function_(x)).first;
    }
    return it->second;
  }

  /// Clear the cache entries
  void clear() noexcept { cache_.clear(); }
 private:
  Function function_;
  Map cache_;
};

/// Construct a CachedFunction wrapping the given function
///
/// \example
/// \snippet Test_CachedFunction.cpp make_cached_function_example
///
/// \tparam Input function argument type
/// \tparam Map class template to use as the map holding the cache
/// \param function the function
/// \param map_args arguments to pass to the map constructor
template <typename Input, template <typename...> class Map = std::unordered_map,
          typename... MapArgs, typename Function, typename... PassedMapArgs>
auto make_cached_function(Function function,
                          PassedMapArgs... map_args) noexcept {
  using output = std::result_of_t<Function&(const Input&)>;
  return CachedFunction<Function, Map<std::decay_t<Input>,
                                      std::decay_t<output>,
                                      MapArgs...>>(
      std::move(function), std::forward<PassedMapArgs>(map_args)...);
}
