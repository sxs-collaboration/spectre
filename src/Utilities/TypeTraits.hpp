// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

/// \ingroup TypeTraitsGroup
/// C++ STL code present in C++17
namespace cpp17 {

/*!
 * \ingroup UtilitiesGroup
 * \brief Mark a return type as being "void". In C++17 void is a regular type
 * under certain circumstances, so this can be replaced by `void` then.
 *
 * The proposal is available
 * [here](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0146r1.html)
 */
struct void_type {};
}  // namespace cpp17

/// \ingroup TypeTraitsGroup
/// C++ STL code present in C++20
namespace cpp20 {
/// \ingroup TypeTraitsGroup
template <class T>
struct remove_cvref {
  // clang-tidy use using instead of typedef
  typedef std::remove_cv_t<std::remove_reference_t<T>> type;  // NOLINT
};

/// \ingroup TypeTraitsGroup
template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;
}  // namespace cpp20
