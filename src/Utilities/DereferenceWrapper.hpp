// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function dereference_wrapper

#pragma once

#include <functional>
#include <utility>

/// \ingroup UtilitiesGroup
/// \brief Returns the reference object held by a reference wrapper, if a
/// non-reference_wrapper type is passed in then the object is returned
template <typename T>
decltype(auto) dereference_wrapper(T&& t) {
  return std::forward<T>(t);
}

/// \cond
template <typename T>
T& dereference_wrapper(const std::reference_wrapper<T>& t) {
  return t.get();
}
template <typename T>
T& dereference_wrapper(std::reference_wrapper<T>& t) {
  return t.get();
}
template <typename T>
T&& dereference_wrapper(const std::reference_wrapper<T>&& t) {
  return t.get();
}
template <typename T>
T&& dereference_wrapper(std::reference_wrapper<T>&& t) {
  return t.get();
}
/// \endcond
