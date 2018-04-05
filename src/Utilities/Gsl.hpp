
/// \file
/// Defines functions and classes from the GSL

#pragma once

#pragma GCC system_header

// The code in this file is adapted from Microsoft's GSL that can be found at
// https://github.com/Microsoft/GSL
// The original license and copyright are:
///////////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015 Microsoft Corporation. All rights reserved.
//
// This code is licensed under the MIT License (MIT).
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
///////////////////////////////////////////////////////////////////////////////
// The code changes are because SpECTRE is not allowed to throw under any
// circumstances that cannot be guaranteed to be caught and so all throw's
// are replaced by hard errors (ERROR).

#include <type_traits>

#include "ErrorHandling/ExpectsAndEnsures.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"

#if defined(__clang__) || defined(__GNUC__)

/*!
 * \ingroup UtilitiesGroup
 * The if statement is expected to evaluate true most of the time
 */
#define LIKELY(x) __builtin_expect(!!(x), 1)

/*!
 * \ingroup UtilitiesGroup
 * The if statement is expected to evaluate false most of the time
 */
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

#else
/*!
 * \ingroup UtilitiesGroup
 * The if statement is expected to evaluate true most of the time
 */
#define LIKELY(x) (x)

/*!
 * \ingroup UtilitiesGroup
 * The if statement is expected to evaluate false most of the time
 */
#define UNLIKELY(x) (x)
#endif

/*!
 * \ingroup UtilitiesGroup
 * \brief Implementations from the Guideline Support Library
 */
namespace gsl {

/*!
 * \ingroup UtilitiesGroup
 * \brief Cast `u` to a type `T` where the cast may result in narrowing
 */
template <class T, class U>
SPECTRE_ALWAYS_INLINE constexpr T narrow_cast(U&& u) noexcept {
  return static_cast<T>(std::forward<U>(u));
}

namespace gsl_detail {
template <class T, class U>
struct is_same_signedness
    : public std::integral_constant<bool, std::is_signed<T>::value ==
                                              std::is_signed<U>::value> {};
}  // namespace gsl_detail

/*!
 * \ingroup UtilitiesGroup
 * \brief A checked version of narrow_cast() that ERRORs if the cast changed
 * the value
 */
template <class T, class U>
SPECTRE_ALWAYS_INLINE T narrow(U u) {
  T t = narrow_cast<T>(u);
  if (static_cast<U>(t) != u) {
    ERROR("Failed to cast " << u << " of type " << pretty_type::get_name<U>()
                            << " to type " << pretty_type::get_name<T>());
  }
  if (not gsl_detail::is_same_signedness<T, U>::value and
      ((t < T{}) != (u < U{}))) {
    ERROR("Failed to cast " << u << " of type " << pretty_type::get_name<U>()
                            << " to type " << pretty_type::get_name<T>());
  }
  return t;
}

// @{
/*!
 * \ingroup UtilitiesGroup
 * \brief Retrieve a entry from a container, with checks in Debug mode that
 * the index being retrieved is valid.
 */
template <class T, std::size_t N, typename Size>
SPECTRE_ALWAYS_INLINE constexpr T& at(std::array<T, N>& arr, Size index) {
  Expects(index >= 0 and index < narrow_cast<Size>(N));
  return arr[static_cast<std::size_t>(index)];
}

template <class Cont, typename Size>
SPECTRE_ALWAYS_INLINE constexpr const typename Cont::value_type& at(
    const Cont& cont, Size index) {
  Expects(index >= 0 and index < narrow_cast<Size>(cont.size()));
  return cont[static_cast<typename Cont::size_type>(index)];
}

template <class T, typename Size>
SPECTRE_ALWAYS_INLINE constexpr const T& at(std::initializer_list<T> cont,
                                            Size index) {
  Expects(index >= 0 and index < narrow_cast<Size>(cont.size()));
  return *(cont.begin() + index);
}
// @}

namespace detail {
template <class T>
struct owner_impl {
  static_assert(std::is_same<T, const owner_impl<int*>&>::value,
                "You should not have an owning raw pointer, instead you should "
                "use std::unique_ptr or, sparingly, std::shared_ptr. If "
                "clang-tidy told you to use gsl::owner, then you should still "
                "use std::unique_ptr instead.");
  using type = T;
};
}  // namespace detail

/*!
 * \ingroup UtilitiesGroup
 * \brief Mark a raw pointer as owning its data
 *
 * \warning You should never actually use `gsl::owner`. Instead you should use
 * `std::unique_ptr`, and if shared ownership is required, `std::shared_ptr`.
 */
template <class T, Requires<std::is_pointer<T>::value> = nullptr>
using owner = typename detail::owner_impl<T>::type;

/*!
 * \ingroup UtilitiesGroup
 * \brief Require a pointer to not be a `nullptr`
 *
 * Restricts a pointer or smart pointer to only hold non-null values.
 *
 * Has zero size overhead over `T`.
 *
 * If `T` is a pointer (i.e. `T == U*`) then
 * - allow construction from `U*`
 * - disallow construction from `nullptr_t`
 * - disallow default construction
 * - ensure construction from null `U*` fails
 * - allow implicit conversion to `U*`
 */
template <class T>
class not_null {
 public:
  static_assert(std::is_assignable<T&, std::nullptr_t>::value,
                "T cannot be assigned nullptr.");

  template <typename U, Requires<std::is_convertible<U, T>::value> = nullptr>
  constexpr not_null(U&& u) : ptr_(std::forward<U>(u)) {
    Expects(ptr_ != nullptr);
  }

  template <typename U, Requires<std::is_convertible<U, T>::value> = nullptr>
  constexpr not_null(const not_null<U>& other) : not_null(other.get()) {}

  not_null(const not_null& other) = default;
  not_null& operator=(const not_null& other) = default;

  constexpr T get() const {
    Ensures(ptr_ != nullptr);
    return ptr_;
  }

  constexpr operator T() const { return get(); }
  constexpr T operator->() const { return get(); }
  constexpr decltype(auto) operator*() const { return *get(); }

  // prevents compilation when someone attempts to assign a null pointer
  // constant
  not_null(std::nullptr_t) = delete;
  not_null& operator=(std::nullptr_t) = delete;

  // unwanted operators...pointers only point to single objects!
  not_null& operator++() = delete;
  not_null& operator--() = delete;
  not_null operator++(int) = delete;
  not_null operator--(int) = delete;
  not_null& operator+=(std::ptrdiff_t) = delete;
  not_null& operator-=(std::ptrdiff_t) = delete;
  void operator[](std::ptrdiff_t) const = delete;

 private:
  T ptr_;
};

template <class T>
std::ostream& operator<<(std::ostream& os, const not_null<T>& val) {
  os << val.get();
  return os;
}

template <class T, class U>
auto operator==(const not_null<T>& lhs, const not_null<U>& rhs)
    -> decltype(lhs.get() == rhs.get()) {
  return lhs.get() == rhs.get();
}

template <class T, class U>
auto operator!=(const not_null<T>& lhs, const not_null<U>& rhs)
    -> decltype(lhs.get() != rhs.get()) {
  return lhs.get() != rhs.get();
}

template <class T, class U>
auto operator<(const not_null<T>& lhs, const not_null<U>& rhs)
    -> decltype(lhs.get() < rhs.get()) {
  return lhs.get() < rhs.get();
}

template <class T, class U>
auto operator<=(const not_null<T>& lhs, const not_null<U>& rhs)
    -> decltype(lhs.get() <= rhs.get()) {
  return lhs.get() <= rhs.get();
}

template <class T, class U>
auto operator>(const not_null<T>& lhs, const not_null<U>& rhs)
    -> decltype(lhs.get() > rhs.get()) {
  return lhs.get() > rhs.get();
}

template <class T, class U>
auto operator>=(const not_null<T>& lhs, const not_null<U>& rhs)
    -> decltype(lhs.get() >= rhs.get()) {
  return lhs.get() >= rhs.get();
}

// more unwanted operators
template <class T, class U>
std::ptrdiff_t operator-(const not_null<T>&, const not_null<U>&) = delete;
template <class T>
not_null<T> operator-(const not_null<T>&, std::ptrdiff_t) = delete;
template <class T>
not_null<T> operator+(const not_null<T>&, std::ptrdiff_t) = delete;
template <class T>
not_null<T> operator+(std::ptrdiff_t, const not_null<T>&) = delete;

}  // namespace gsl

// The remainder of this file is
// Distributed under the MIT License.
// See LICENSE.txt for details.

/// Construct a not_null from a pointer.  Often this will be done as
/// an implicit conversion, but it may be necessary to perform the
/// conversion explicitly when type deduction is desired.
///
/// \note This is not a standard GSL function, and so is not in the
/// gsl namespace.
template <typename T>
gsl::not_null<T*> make_not_null(T* ptr) noexcept {
  return gsl::not_null<T*>(ptr);
}
