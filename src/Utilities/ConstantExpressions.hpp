// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Define simple functions for constant expressions.

#pragma once

#include <algorithm>
#include <blaze/math/typetraits/IsVector.h>
#include <type_traits>

#include "ErrorHandling/Assert.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// \ingroup ConstantExpressions
/// Compute 2 to the n for integral types.
///
/// \param n the power of two to compute.
/// \return 2^n
template <typename T,
          Requires<tt::is_integer_v<T> and cpp17::is_unsigned_v<T>> = nullptr>
SPECTRE_ALWAYS_INLINE constexpr T two_to_the(T n) {
  return T(1) << n;
}

/// \ingroup ConstantExpressions
/// Get the nth bit from the right, counting from zero.
///
/// \param i the value to be acted on.
/// \param N which place to extract the bit
/// \return the value of the bit at that place.
SPECTRE_ALWAYS_INLINE constexpr size_t get_nth_bit(const size_t i,
                                                   const size_t N) {
  return (i >> N) % 2;
}

/// \ingroup ConstantExpressions
/// \brief Compute the square of `x`
template <typename T>
SPECTRE_ALWAYS_INLINE constexpr decltype(auto) square(const T& x) {
  return x * x;
}

/// \ingroup ConstantExpressions
/// \brief Compute the cube of `x`
template <typename T>
SPECTRE_ALWAYS_INLINE constexpr decltype(auto) cube(const T& x) {
  return x * x * x;
}

/// \ingroup ConstantExpressions
namespace ConstantExpressions_detail {
template <typename T, int N, typename = std::nullptr_t>
struct pow;

/// \cond HIDDEN_SYMBOLS
template <typename T, int N>
struct pow<T, N, Requires<(N > 0)>> {
  SPECTRE_ALWAYS_INLINE static constexpr decltype(auto) apply(const T& t) {
    return t * pow<T, N - 1>::apply(t);
  }
};

template <typename T, int N>
struct pow<T, N, Requires<(N < 0)>> {
  SPECTRE_ALWAYS_INLINE static constexpr decltype(auto) apply(const T& t) {
    return static_cast<T>(1) / (t * pow<T, -N - 1>::apply(t));
  }
};

template <typename T>
struct pow<T, 0, Requires<not blaze::IsVector<T>::value>> {
  SPECTRE_ALWAYS_INLINE static constexpr T apply(const T& /*t*/) {
    return static_cast<T>(1);
  }
};
/// \endcond
}  // namespace ConstantExpressions_detail

/// \ingroup ConstantExpressions
/// \brief Compute t^N where N is an integer (positive or negative)
///
/// \warning If passing an integer this will do integer arithmatic, e.g.
/// pow<-10>(2) == 0
///
/// \tparam N the integer power being raised to in t^N
/// \param t the value being exponentiated
/// \return value t^N of type T
template <int N, typename T>
SPECTRE_ALWAYS_INLINE constexpr decltype(auto) pow(const T& t) {
  return ConstantExpressions_detail::pow<T, N>::apply(t);
}

/// \ingroup ConstantExpressions
/// \brief Compute the absolute value of of its argument
///
/// The argument must be comparable to an int and muste be negatable.
template <typename T>
constexpr T constexpr_abs(const T& x) noexcept(noexcept(x < 0 ? -x : x)) {
  return x < 0 ? -x : x;
}

namespace ConstantExpressions_detail {
struct CompareByMagnitude {
  template <typename T>
  constexpr bool operator()(const T& a, const T& b) {
    return constexpr_abs(a) < constexpr_abs(b);
  }
};
}  // namespace ConstantExpressions_detail

/// \ingroup ConstantExpressions
/// \brief Return the argument with the largest magnitude
///
/// Magnitude is determined by constexpr_abs.  In case of a tie,
/// returns the leftmost of the tied values.
//@{
template <typename T>
constexpr const T& max_by_magnitude(const T& a, const T& b) {
  return std::max(a, b, ConstantExpressions_detail::CompareByMagnitude{});
}

template <typename T>
constexpr T max_by_magnitude(std::initializer_list<T> ilist) {
  return std::max(std::move(ilist),
                  ConstantExpressions_detail::CompareByMagnitude{});
}
//@}

/// \ingroup ConstantExpressions
/// \brief Return the argument with the smallest magnitude
///
/// Magnitude is determined by constexpr_abs.  In case of a tie,
/// returns the leftmost of the tied values.
//@{
template <typename T>
constexpr const T& min_by_magnitude(const T& a, const T& b) {
  return std::min(a, b, ConstantExpressions_detail::CompareByMagnitude{});
}

template <typename T>
constexpr T min_by_magnitude(std::initializer_list<T> ilist) {
  return std::min(std::move(ilist),
                  ConstantExpressions_detail::CompareByMagnitude{});
}
//@}

namespace detail {
template <typename List, size_t... indices,
          Requires<not tt::is_a_v<tmpl::list, tmpl::front<List>>> = nullptr>
inline constexpr std::array<std::decay_t<decltype(tmpl::front<List>::value)>,
                            tmpl::size<List>::value>
make_array_from_list_helper(
    std::integer_sequence<size_t, indices...> /*meta*/) {
  return std::array<std::decay_t<decltype(tmpl::front<List>::value)>,
                    tmpl::size<List>::value>{
      {tmpl::at<List, tmpl::size_t<indices>>::value...}};
}
}  // namespace detail

/// \ingroup ConstantExpressions
/// Make an array from a typelist that holds std::integral_constant's all of
/// which have the same `value_type`
///
/// \tparam List the typelist of std::integral_constant's
/// \return array of integral values from the typelist
template <typename List,
          Requires<not tt::is_a_v<tmpl::list, tmpl::front<List>>> = nullptr>
inline constexpr std::array<std::decay_t<decltype(tmpl::front<List>::value)>,
                            tmpl::size<List>::value>
make_array_from_list() {
  return detail::make_array_from_list_helper<List>(
      std::make_integer_sequence<size_t, tmpl::size<List>::value>{});
}

template <typename TypeForZero,
          Requires<not tt::is_a_v<tmpl::list, TypeForZero>> = nullptr>
inline constexpr std::array<std::decay_t<TypeForZero>, 0>
make_array_from_list() {
  return std::array<std::decay_t<TypeForZero>, 0>{{}};
}

namespace detail {
template <typename List, size_t... indices,
          Requires<tt::is_a<tmpl::list, tmpl::front<List>>::value> = nullptr>
inline constexpr std::array<
    std::decay_t<
        decltype(make_array_from_list<tmpl::at<List, tmpl::size_t<0>>>())>,
    tmpl::size<List>::value>
make_array_from_list_helper(
    std::integer_sequence<size_t, indices...> /*unused*/) {
  return std::array<std::decay_t<decltype(make_array_from_list<
                                          tmpl::at<List, tmpl::size_t<0>>>())>,
                    tmpl::size<List>::value>{
      {make_array_from_list_helper<tmpl::at<List, tmpl::size_t<indices>>>(
          std::make_integer_sequence<
              size_t,
              tmpl::size<tmpl::at<List, tmpl::size_t<indices>>>::value>{})...}};
}
}  // namespace detail

/// \ingroup ConstantExpressions
///
/// Make an array of arrays from a typelist that holds typelists of
/// std::integral_constant's all of which have the same `value_type`
///
/// \tparam List the typelist of typelists of std::integral_constant's
/// \return array of arrays of integral values from the typelists
template <typename List,
          Requires<tt::is_a_v<tmpl::list, tmpl::front<List>>> = nullptr>
inline constexpr auto make_array_from_list() {
  return detail::make_array_from_list_helper<List>(
      std::make_integer_sequence<size_t, tmpl::size<List>::value>{});
}

/// \ingroup ConstantExpressions
/// \brief Compute the length of a const char* at compile time
SPECTRE_ALWAYS_INLINE constexpr size_t cstring_length(
    const char* str) noexcept {
  // clang-tidy: do not use pointer arithmetic
  return *str != 0 ? 1 + cstring_length(str + 1) : 0;  // NOLINT
}

/// \ingroup ConstantExpressions
/// \brief Compute a hash of a const char* at compile time
SPECTRE_ALWAYS_INLINE constexpr size_t cstring_hash(const char* str) noexcept {
  // clang-tidy: do not use pointer arithmetic
  return *str != 0
             ? (cstring_hash(str + 1) * 33) ^  // NOLINT
                   static_cast<size_t>(*str)
             : 5381;
}

namespace ConstantExpression_detail {
template <typename T, size_t Size, size_t... I, size_t... J>
inline constexpr std::array<std::decay_t<T>, Size> replace_at_helper(
    const std::array<T, Size>& arr, const T& value, const size_t i,
    std::integer_sequence<size_t, I...> /*unused*/,
    std::integer_sequence<size_t, J...> /*unused*/) {
  // clang-tidy: Cannot use gsl::at because we want constexpr evaluation and
  // Parallel::abort violates this
  return std::array<std::decay_t<T>, Size>{
      {arr[I]..., value, arr[i + J]...}};  // NOLINT
}
}  // namespace ConstantExpression_detail

/// \ingroup ConstantExpressions
/// Replace at compile time the `I`th entry in the array with `value`
template <size_t I, typename T, size_t Size>
inline constexpr std::array<std::decay_t<T>, Size> replace_at(
    const std::array<T, Size>& arr, T value) {
  return ConstantExpression_detail::replace_at_helper(
      arr, std::forward<T>(value), I + 1,
      std::make_integer_sequence<size_t, I>{},
      std::make_integer_sequence<size_t, Size - I - 1>{});
}

/// \ingroup ConstantExpressions
/// Check at compile time if two `std::array`s are equal
template <typename T, typename S, size_t size>
inline constexpr bool array_equal(const std::array<T, size>& lhs,
                                  const std::array<S, size>& rhs,
                                  const size_t i = 0) noexcept {
  // clang-tidy: Cannot use gsl::at because we want constexpr evaluation and
  // Parallel::abort violates this
  return i < size ? (lhs[i] == rhs[i]  // NOLINT
                     and array_equal(lhs, rhs, i + 1))
                  : true;
}

/// \ingroup ConstantExpressions
/// \brief Returns a const reference to its argument.
template <typename T>
constexpr const T& as_const(const T& t) noexcept { return t; }
