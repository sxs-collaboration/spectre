// Distributed under the MIT License.
// See LICENSE.txt for details.
#pragma once

#include <array>
#include <cstddef>
#include <memory>  // IWYU pragma: keep
#include <tuple>
#include <utility>

#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/Tuple.hpp"
#include "tests/Unit/TestingFramework.hpp"

namespace TestHelpers {
namespace VectorImpl {

namespace detail {
// `check_vectors` checks equality of vectors and related data types for tests.
//
// Comparisons between any pairs of the following are supported:
// vector types
// arithmetic types
// arrays of vectors
// arrays of arithmetic types

// between two vectors
template <typename T1, typename T2,
          Requires<cpp17::is_arithmetic_v<typename T1::ElementType> and
                   cpp17::is_arithmetic_v<typename T2::ElementType>> = nullptr>
inline void check_vectors(const T1& t1, const T2& t2) noexcept {
  CHECK_ITERABLE_APPROX(typename T1::ResultType{t1},
                        typename T2::ResultType{t2});
}
// between two arithmetic types
template <typename T1, typename T2,
          Requires<cpp17::is_arithmetic_v<T1> and cpp17::is_arithmetic_v<T2>> =
              nullptr>
inline void check_vectors(const T1& t1, const T2& t2) noexcept {
  CHECK(approx(t1) == t2);
}
// between an arithmetic type and a vector
template <typename T1, typename T2,
          Requires<cpp17::is_arithmetic_v<T1> and
                   cpp17::is_arithmetic_v<typename T2::ElementType>> = nullptr>
void check_vectors(const T1& t1, const T2& t2) noexcept {
  check_vectors(typename T2::ResultType{t2.size(), t1}, t2);
}
// between a vector and an arithmetic type
template <typename T1, typename T2,
          Requires<cpp17::is_arithmetic_v<typename T1::ElementType> and
                   cpp17::is_arithmetic_v<T2>> = nullptr>
void check_vectors(const T1& t1, const T2& t2) noexcept {
  check_vectors(t2, t1);
}
// between two arrays
template <typename T1, typename T2, size_t S>
void check_vectors(const std::array<T1, S>& t1,
                   const std::array<T2, S>& t2) noexcept {
  for (size_t i = 0; i < S; i++) {
    check_vectors(gsl::at(t1, i), gsl::at(t2, i));
  }
}
// between an array of vectors and an arithmetic type
template <typename T1, typename T2, size_t S,
          Requires<cpp17::is_arithmetic_v<T2>> = nullptr,
          Requires<cpp17::is_arithmetic_v<typename T1::ElementType>> = nullptr>
void check_vectors(const std::array<T1, S>& t1, const T2& t2) noexcept {
  std::array<T1, S> compArr;
  compArr.fill(typename T1::ResultType(t1[0].size(), t2));
  check_vectors(t1, compArr);
}
// between an array of vectors and a vector
template <typename T1, typename T2, size_t S,
          Requires<cpp17::is_arithmetic_v<typename T2::ElementType>> = nullptr,
          Requires<cpp17::is_arithmetic_v<typename T1::ElementType>> = nullptr>
void check_vectors(const std::array<T1, S>& t1, const T2& t2) noexcept {
  std::array<T1, S> compArr;
  compArr.fill(typename T1::ResultType(t2));
  check_vectors(t1, compArr);
}
// between an arithmetic type and an array of vectors
template <typename T1, typename T2, size_t S,
          Requires<cpp17::is_arithmetic_v<T1>> = nullptr,
          Requires<cpp17::is_arithmetic_v<typename T2::ElementType>> = nullptr>
void check_vectors(const T1& t1, const std::array<T2, S>& t2) noexcept {
  check_vectors(t2, t1);
}
// between a vector and an array of vectors
template <typename T1, typename T2, size_t S,
          Requires<cpp17::is_arithmetic_v<typename T1::ElementType>> = nullptr,
          Requires<cpp17::is_arithmetic_v<typename T2::ElementType>> = nullptr>
void check_vectors(const T1& t1, const std::array<T2, S>& t2) noexcept {
  check_vectors(t2, t1);
}
} // namespace detail

// @{
/*!
 * \brief Perform a function over combinatoric subsets of a `std::tuple`
 *
 * \details
 * If given a function object, the apply_tuple_combinations function will
 * apply the function to all ordered combinations. For instance, if the tuple
 * contains elements (a,b,c) and the function f accepts two arguments it will be
 * executed on all nine combinations of the three possible
 * arguments. Importantly, the function f must then be invokable for all such
 * argument combinations.
 *
 * \tparam ArgLength the number of arguments taken by the invokable. In the
 * case where `function` is an object or closure whose call operator is
 * overloaded or a template it is not possible to determine the number of
 * arguments using a type trait. One common use case would be a generic
 * lambda. For this reason it is necessary to explicitly specify the number of
 * arguments to the invokable.
 *
 * \param set - the tuple for which you'd like to execute every combination
 *
 * \param function - The a callable, which is called on the combinations of the
 * tuple
 *
 * \param args - a set of partially assembled arguments passed recursively. Some
 * arguments can be passed in by the using code, in which case those are kept at
 * the end of the argument list and the combinations are prependend.
 */
template <size_t ArgLength, typename T, typename... Elements, typename... Args,
          Requires<sizeof...(Args) == ArgLength> = nullptr>
constexpr inline void apply_tuple_combinations(
    const std::tuple<Elements...>& set, const T& function,
    const Args&... args) noexcept {
  (void)
      set;  // so it is used, can be documented, and causes no compiler warnings
  function(args...);
}

template <size_t ArgLength, typename T, typename... Elements, typename... Args,
          Requires<(ArgLength > sizeof...(Args))> = nullptr>
constexpr inline void apply_tuple_combinations(
    const std::tuple<Elements...>& set, const T& function,
    const Args&... args) noexcept {
  tuple_fold(set, [&function, &set, &args...](const auto& x) noexcept {
      apply_tuple_combinations<ArgLength>(set, function, x, args...);
  });
}
// @}

namespace detail {
template <typename... ValueTypes, size_t... Is>
auto addressof_impl(gsl::not_null<std::tuple<ValueTypes...>*> preset,
                    std::index_sequence<Is...> /*meta*/) noexcept {
  return std::make_tuple(std::addressof(std::get<Is>(*preset))...);
}

template <size_t N, typename... ValueTypes, size_t... Is1, size_t... Is2>
auto remove_nth_impl(const std::tuple<ValueTypes...>& tup,
                     std::index_sequence<Is1...> /*meta*/,
                     std::index_sequence<Is2...> /*meta*/) noexcept {
  return std::make_tuple(std::get<Is1>(tup)..., std::get<N + Is2 + 1>(tup)...);
}
}  // namespace detail

/*!
 * \brief given a pointer to a tuple, returns a tuple filled with pointers to
 * the given tuple's elements
 */
template <typename... ValueTypes>
auto addressof(gsl::not_null<std::tuple<ValueTypes...>*> preset) noexcept {
  return detail::addressof_impl(
      preset, std::make_index_sequence<sizeof...(ValueTypes)>());
}

/*!
 * \brief given a tuple, returns a tuple with the specified element removed
 *
 * \tparam N the element to remove
 *
 * \param tup the tuple from which to remove the element
 */
template <size_t N, typename... ValueTypes>
auto remove_nth(const std::tuple<ValueTypes...>& tup) noexcept {
  return detail::remove_nth_impl<N>(
      tup, std::make_index_sequence<N>(),
      std::make_index_sequence<sizeof...(ValueTypes) - N - 1>());
}
}  // namespace VectorImpl
}  // namespace TestHelpers
