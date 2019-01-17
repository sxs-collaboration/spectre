// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Define simple functions for constant expressions.

#pragma once

#include <algorithm>
#include <array>
#include <blaze/math/typetraits/IsVector.h>  // IWYU pragma: keep
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <utility>

#include "Utilities/ForceInline.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

// IWYU pragma: no_include "DataStructures/DataVector.hpp"

/// \ingroup ConstantExpressionsGroup
/// Compute 2 to the n for integral types.
///
/// \param n the power of two to compute.
/// \return 2^n
template <typename T,
          Requires<tt::is_integer_v<T> and cpp17::is_unsigned_v<T>> = nullptr>
SPECTRE_ALWAYS_INLINE constexpr T two_to_the(T n) {
  return T(1) << n;
}

/// \ingroup ConstantExpressionsGroup
/// Get the nth bit from the right, counting from zero.
///
/// \param i the value to be acted on.
/// \param N which place to extract the bit
/// \return the value of the bit at that place.
SPECTRE_ALWAYS_INLINE constexpr size_t get_nth_bit(const size_t i,
                                                   const size_t N) {
  return (i >> N) % 2;
}

/// \ingroup ConstantExpressionsGroup
/// \brief Compute the square of `x`
template <typename T>
SPECTRE_ALWAYS_INLINE constexpr decltype(auto) square(const T& x) {
  return x * x;
}

/// \ingroup ConstantExpressionsGroup
/// \brief Compute the cube of `x`
template <typename T>
SPECTRE_ALWAYS_INLINE constexpr decltype(auto) cube(const T& x) {
  return x * x * x;
}

/*!
 * \ingroup ConstantExpressionsGroup
 * \brief Compute the falling factorial of \f$(x)_{n}\f$
 *
 * See http://mathworld.wolfram.com/FallingFactorial.html
 * \note The largest representable factorial is 20!. It is up to the user to
 * ensure this is satisfied
 */
constexpr uint64_t falling_factorial(const uint64_t x,
                                     const uint64_t n) noexcept {
  // clang-tidy: don't warn about STL internals, I can't fix them
  assert(n <= x);  // NOLINT
  uint64_t r = 1;
  for (uint64_t k = 0; k < n; ++k) {
    r *= (x - k);
  }
  return r;
}

/*!
 * \ingroup ConstantExpressionsGroup
 * \brief Compute the factorial of \f$n!\f$
 */
constexpr uint64_t factorial(const uint64_t n) noexcept {
  assert(n <= 20);  // NOLINT
  return falling_factorial(n, n);
}

/// \ingroup ConstantExpressionsGroup
namespace ConstantExpressions_detail {

// Implementation functions for the pow template function below which computes
// optimized powers where the exponent is a compile-time integer

// base case power 0: Returns 1.0. This will need to be overloaded for types for
// which the simple return is inappropriate.
template <typename T>
SPECTRE_ALWAYS_INLINE constexpr decltype(auto) pow_impl(
    const T& /*t*/, std::integral_constant<int, 0> /*meta*/,
    cpp17::bool_constant<true> /*exponent_was_positive*/) noexcept {
  return static_cast<tt::get_fundamental_type_t<T>>(1.0);
}

// special case power 1: acts as a direct identity function for efficiency
template <typename T>
SPECTRE_ALWAYS_INLINE constexpr decltype(auto) pow_impl(
    const T& t, std::integral_constant<int, 1> /*meta*/,
    cpp17::bool_constant<true> /*exponent_was_positive*/) noexcept {
  return t;
}

// general case for positive powers: return the power via recursive inline call,
// which expands to a series of multiplication operations.
template <int N, typename T>
SPECTRE_ALWAYS_INLINE constexpr decltype(auto) pow_impl(
    const T& t, std::integral_constant<int, N> /*meta*/,
    cpp17::bool_constant<true> /*exponent_was_positive*/) noexcept {
  return t * pow_impl(t, std::integral_constant<int, N - 1>{},
                      cpp17::bool_constant<true>{});
}

// general case for negative powers: return the multiplicative inverse of the
// result from the recursive inline call, which expands to a series of
// multiplication operations. This assumes that division is supported with
// tt::get_fundamental_type_t<T> and T. If not, this utility will need further
// specialization.
template <int N, typename T>
SPECTRE_ALWAYS_INLINE constexpr decltype(auto) pow_impl(
    const T& t, std::integral_constant<int, N> /*meta*/,
    cpp17::bool_constant<false> /*exponent_was_positive*/) noexcept {
  return static_cast<tt::get_fundamental_type_t<T>>(1) /
         (pow_impl(t, std::integral_constant<int, -N>{},
                   cpp17::bool_constant<true>{}));
}
}  // namespace ConstantExpressions_detail

/// \ingroup ConstantExpressionsGroup
/// \brief Compute t^N where N is an integer (positive or negative)
///
/// \warning If passing an integer this will do integer arithmetic, e.g.
/// `pow<-10>(2) == 0` evaluates to `true`
///
/// \warning For optimization, it is assumed that the `pow<0>` of a vector type
/// (e.g. `DataVector`) will not be used for initialization or directly indexed,
/// so `pow<0>` returns simply `1.0`. In the case of use for initialization, a
/// constructor should be used instead, and in the case of a direct index, the
/// expression may be simplifyable to avoid the use of `pow<0>` altogether. If a
/// more complete treatment of `pow<0>` is required, further overloads may be
/// added to the `ConstantExpressions_detail` namespace.
///
/// \tparam N the integer power being raised to in \f$t^N\f$
/// \param t the value being exponentiated
/// \return value \f$t^N\f$ determined via repeated multiplication
template <int N, typename T>
SPECTRE_ALWAYS_INLINE constexpr decltype(auto) pow(const T& t) noexcept {
  return ConstantExpressions_detail::pow_impl(
      t, std::integral_constant<int, N>{}, cpp17::bool_constant<(N >= 0)>{});
}

/// \ingroup ConstantExpressionsGroup
/// \brief Compute the absolute value of of its argument
///
/// The argument must be comparable to an int and must be negatable.
template <typename T, Requires<tt::is_integer_v<T> or
                               cpp17::is_floating_point_v<T>> = nullptr>
SPECTRE_ALWAYS_INLINE constexpr T ce_abs(const T& x) noexcept(
    noexcept(x < 0 ? -x : x)) {
  return x < 0 ? -x : x;
}

/// \cond
template <>
SPECTRE_ALWAYS_INLINE constexpr double ce_abs(const double& x) noexcept {
  return __builtin_fabs(x);
}

template <>
SPECTRE_ALWAYS_INLINE constexpr float ce_abs(const float& x) noexcept {
  return __builtin_fabsf(x);
}
/// \endcond

/// \ingroup ConstantExpressionsGroup
/// \brief Compute the absolute value of its argument
constexpr SPECTRE_ALWAYS_INLINE double ce_fabs(const double x) noexcept {
  return __builtin_fabs(x);
}

constexpr SPECTRE_ALWAYS_INLINE float ce_fabs(const float x) noexcept {
  return __builtin_fabsf(x);
}

namespace ConstantExpressions_detail {
struct CompareByMagnitude {
  template <typename T>
  constexpr bool operator()(const T& a, const T& b) {
    return ce_abs(a) < ce_abs(b);
  }
};
}  // namespace ConstantExpressions_detail

/// \ingroup ConstantExpressionsGroup
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

/// \ingroup ConstantExpressionsGroup
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

namespace cpp17 {
/// \ingroup ConstantExpressionsGroup
/// \brief Clamps the value between lo and hi
///
/// If v compares less than lo, returns lo; otherwise if hi compares less than
///  v, returns hi; otherwise returns v.
template <class T, class Compare = std::less<>>
constexpr const T& clamp(const T& v, const T& lo, const T& hi,
                         Compare comp = Compare()) {
  // reason for NOLINT: the warning below occurs despite no instances of an
  // array in the clamp calls. This warning occurs sometime during the assert
  // macro expansion rather than being due to an implementation error.
  // "warning: do not implicitly decay an array into a pointer; consider using
  //  gsl::array_view or an explicit cast instead
  //  [cppcoreguidelines-pro-bounds-array-to-pointer-decay]"
  return assert(!comp(hi, lo)),  // NOLINT
         comp(v, lo) ? lo : comp(hi, v) ? hi : v;
}
}  // namespace cpp17

/// \ingroup ConstantExpressionsGroup
/// \brief Returns `f(ic<0>{}) + f(ic<1>{}) + ... + f(ic<NumTerms-1>{})`
/// where `ic<N>` stands for `std::integral_constant<size_t, N>`.
/// This function allows the result types of each operation to be
/// different, and so works efficiently with expression templates.
/// \note When summing expression templates one must be careful of
/// referring to temporaries in `f`.
template <size_t NumTerms, typename Function, Requires<NumTerms == 1> = nullptr>
constexpr decltype(auto) constexpr_sum(Function&& f) noexcept {
  return f(std::integral_constant<size_t, 0>{});
}

/// \cond HIDDEN_SYMBOLS
template <size_t NumTerms, typename Function,
          Requires<(NumTerms > 1)> = nullptr>
constexpr decltype(auto) constexpr_sum(Function&& f) noexcept {
  return constexpr_sum<NumTerms - 1>(f) +
         f(std::integral_constant<size_t, NumTerms - 1>{});
}
/// \endcond

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

/// \ingroup ConstantExpressionsGroup
/// Make an array from a typelist that holds std::integral_constant's all of
/// which have the same `value_type`
///
/// \tparam List the typelist of std::integral_constant's
/// \return array of integral values from the typelist
template <typename List,
          Requires<not tt::is_a_v<tmpl::list, tmpl::front<List>>> = nullptr>
inline constexpr auto make_array_from_list()
    -> std::array<std::decay_t<decltype(tmpl::front<List>::value)>,
                  tmpl::size<List>::value> {
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

/// \ingroup ConstantExpressionsGroup
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

/// \ingroup ConstantExpressionsGroup
/// \brief Compute the length of a const char* at compile time
SPECTRE_ALWAYS_INLINE constexpr size_t cstring_length(
    const char* str) noexcept {
  // clang-tidy: do not use pointer arithmetic
  return *str != 0 ? 1 + cstring_length(str + 1) : 0;  // NOLINT
}

/// \ingroup ConstantExpressionsGroup
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

/// \ingroup ConstantExpressionsGroup
/// Replace at compile time the `I`th entry in the array with `value`
template <size_t I, typename T, size_t Size>
inline constexpr std::array<std::decay_t<T>, Size> replace_at(
    const std::array<T, Size>& arr, T value) {
  return ConstantExpression_detail::replace_at_helper(
      arr, std::forward<T>(value), I + 1,
      std::make_integer_sequence<size_t, I>{},
      std::make_integer_sequence<size_t, Size - I - 1>{});
}

/// \ingroup ConstantExpressionsGroup
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

namespace cpp17 {
/// \ingroup ConstantExpressionsGroup
/// \brief Returns a const reference to its argument.
template <typename T>
constexpr const T& as_const(const T& t) noexcept {
  return t;
}
}  // namespace cpp17
