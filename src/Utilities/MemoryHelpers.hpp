// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>

namespace cpp20 {
namespace detail {
template <typename T>
struct unique_type {
  using single_object = std::unique_ptr<T>;
};

template <typename T>
struct unique_type<T[]> {
  using array = std::unique_ptr<T[]>;
};

template<typename T, size_t Bound>
struct unique_type<T[Bound]> { struct invalid_type { }; };
}  // namespace detail

template <typename T, typename... Args>
typename detail::unique_type<T>::single_object make_unique_for_overwrite() {
  return std::unique_ptr<T>(new T);
}

template <typename T>
typename detail::unique_type<T>::array make_unique_for_overwrite(
    const size_t num) {
  return std::unique_ptr<T>(new std::remove_extent_t<T>[num]);
}

template <typename T, typename... Args>
typename detail::unique_type<T>::invalid_type make_unique_for_overwrite(
    Args&&...) = delete;
}  // namespace cpp20
