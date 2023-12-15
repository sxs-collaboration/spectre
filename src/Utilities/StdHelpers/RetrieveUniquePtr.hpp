// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

namespace StdHelpers {
/// @{
/// \brief Dereference a `std::unique_ptr` or just get the value back.
template <typename T, typename Deleter>
const T& retrieve(const std::unique_ptr<T, Deleter>& t) {
  return *t;
}

template <typename T>
const T& retrieve(const T& t) {
  return t;
}
/// @}
}  // namespace StdHelpers
