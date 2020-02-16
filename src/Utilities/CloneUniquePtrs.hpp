// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

/// \ingroup UtilitiesGroup
/// \brief Given a map of `std::unique_ptr` returns a copy of the map by
/// invoking `get_clone()` on each element of the input map.
template <typename KeyType, typename T>
std::unordered_map<KeyType, std::unique_ptr<T>> clone_unique_ptrs(
    const std::unordered_map<KeyType, std::unique_ptr<T>>& map) noexcept {
  std::unordered_map<KeyType, std::unique_ptr<T>> result{};
  for (const auto& kv : map) {
    result[kv.first] = kv.second->get_clone();
  }
  return result;
}

/// \ingroup UtilitiesGroup
/// \brief Given a vector of `std::unique_ptr` returns a copy of the vector by
/// invoking `get_clone()` on each element of the input vector.
template <typename T>
std::vector<std::unique_ptr<T>> clone_unique_ptrs(
    const std::vector<std::unique_ptr<T>>& vector) noexcept {
  std::vector<std::unique_ptr<T>> result{vector.size()};
  for (size_t i = 0; i < vector.size(); ++i) {
    result[i] = vector[i]->get_clone();
  }
  return result;
}
