// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Define simple functions for constant expressions.

#pragma once

#include <type_traits>

#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace blaze {
struct Expression;
}  // namespace blaze

/// \ingroup ConstantExpressions
/// Compute 2 to the n for integral types.
///
/// \param n the power of two to compute.
/// \return 2^n
template <typename T,
          Requires<std::is_integral<T>::value and
                   not std::is_same<bool, T>::value> = nullptr>
SPECTRE_ALWAYS_INLINE constexpr T two_to_the(T n) {
  return static_cast<T>(1) << n;
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
SPECTRE_ALWAYS_INLINE constexpr T square(const T& x) {
  return x * x;
}

/// \ingroup ConstantExpressions
/// \brief Compute the cube of `x`
template <typename T>
SPECTRE_ALWAYS_INLINE constexpr T cube(const T& x) {
  return x * x * x;
}

/// \ingroup ConstantExpressions
namespace ConstantExpressions_details {
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
struct pow<T, 0, Requires<not std::is_base_of<blaze::Expression, T>::value>> {
  SPECTRE_ALWAYS_INLINE static constexpr T apply(const T& /*t*/) {
    return static_cast<T>(1);
  }
};
/// \endcond
}  // namespace ConstantExpressions_details

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
  return ConstantExpressions_details::pow<T, N>::apply(t);
}

namespace detail {
template <typename Ls, size_t... indices,
          Requires<not tt::is_a_v<tmpl::list, tmpl::front<Ls>>> = nullptr>
inline constexpr std::array<std::decay_t<decltype(tmpl::front<Ls>::value)>,
                            tmpl::size<Ls>::value>
make_array_from_list_helper(
    std::integer_sequence<size_t, indices...> /*meta*/) {
  return std::array<std::decay_t<decltype(tmpl::front<Ls>::value)>,
                    tmpl::size<Ls>::value>{
      {tmpl::at<Ls, tmpl::size_t<indices>>::value...}};
}
}  // namespace detail

/// \ingroup ConstantExpressions
/// Make an array from a typelist that holds std::integral_constant's all of
/// which have the same `value_type`
///
/// \tparam Ls the typelist of std::integral_constant's
/// \return array of integral values from the typelist
template <typename Ls,
          Requires<not tt::is_a_v<tmpl::list, tmpl::front<Ls>>> = nullptr>
inline constexpr std::array<std::decay_t<decltype(tmpl::front<Ls>::value)>,
                            tmpl::size<Ls>::value>
make_array_from_list() {
  return detail::make_array_from_list_helper<Ls>(
      std::make_integer_sequence<size_t, tmpl::size<Ls>::value>{});
}

template <typename TypeForZero,
          Requires<not tt::is_a_v<tmpl::list, TypeForZero>> = nullptr>
inline constexpr std::array<std::decay_t<TypeForZero>, 0>
make_array_from_list() {
  return std::array<std::decay_t<TypeForZero>, 0>{{}};
}

namespace detail {
template <typename Ls, size_t... indices,
          Requires<tt::is_a<tmpl::list, tmpl::front<Ls>>::value> = nullptr>
inline constexpr std::array<
    std::decay_t<
        decltype(make_array_from_list<tmpl::at<Ls, tmpl::size_t<0>>>())>,
    tmpl::size<Ls>::value>
make_array_from_list_helper(
    std::integer_sequence<size_t, indices...> /*unused*/) {
  return std::array<std::decay_t<decltype(
                        make_array_from_list<tmpl::at<Ls, tmpl::size_t<0>>>())>,
                    tmpl::size<Ls>::value>{
      {make_array_from_list_helper<tmpl::at<Ls, tmpl::size_t<indices>>>(
          std::make_integer_sequence<
              size_t,
              tmpl::size<tmpl::at<Ls, tmpl::size_t<indices>>>::value>{})...}};
}
}  // namespace detail

/// \ingroup ConstantExpressions
///
/// Make an array of arrays from a typelist that holds typelists of
/// std::integral_constant's all of which have the same `value_type`
///
/// \tparam Ls the typelist of typelists of std::integral_constant's
/// \return array of arrays of integral values from the typelists
template <typename Ls,
          Requires<tt::is_a_v<tmpl::list, tmpl::front<Ls>>> = nullptr>
inline constexpr auto make_array_from_list() {
  return detail::make_array_from_list_helper<Ls>(
      std::make_integer_sequence<size_t, tmpl::size<Ls>::value>{});
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
