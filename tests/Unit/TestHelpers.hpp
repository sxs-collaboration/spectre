// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Commonly used routines, functions and definitions shared amongst unit tests

#pragma once

#include <array>
#include <catch.hpp>
#include <charm++.h>
#include <csignal>
#include <limits>
#include <pup.h>

#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Parallel/Abort.hpp"
#include "Parallel/Serialize.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TypeTraits.hpp"

#define SPECTRE_TEST_CASE(m, n) TEST_CASE(m, n)  // NOLINT

/*!
 * \ingroup TestingFramework
 * \brief Set a default tolerance for floating-point number comparison
 *
 * \details
 * Catch's default (relative) tolerance for comparing floating-point numbers is
 * `std\:\:numeric_limits<float>\:\:epsilon() * 100`, or roughly \f$10^{-5}\f$.
 * This tolerance is too loose for checking many scientific algorithms that
 * rely on double precision floating-point accuracy, so we provide a tighter
 * tighter tolerance through the `approx` static object.
 *
 * \example
 * \snippet TestFramework.cpp approx_test
 */
// clang-tidy: static object creation may throw exception
static Approx approx = Approx::custom().epsilon(    // NOLINT
    std::numeric_limits<double>::epsilon() * 100);  // NOLINT

/// \cond HIDDEN_SYMBOLS
[[noreturn]] inline void spectre_testing_signal_handler(int /*signal*/) {
  Parallel::abort("");
}
/// \endcond

/*!
 * \ingroup TestingFramework
 * \brief Mark a test as checking a call to ERROR
 *
 * \details
 * In order to properly handle aborting with Catch versions newer than 1.6.1
 * we must install a signal handler after Catch does, which means inside the
 * SPECTRE_TEST_CASE itself. The ERROR_TEST() macro should be the first line in
 * the SPECTRE_TEST_CASE.
 *
 * \example
 * \snippet TestFramework.cpp error_test
 */
#define ERROR_TEST()                                      \
  do {                                                    \
    std::signal(SIGABRT, spectre_testing_signal_handler); \
  } while (false)

/*!
 * \ingroup TestingFramework
 * \brief Mark a test to be checking an ASSERT
 *
 * \details
 * Testing error handling is just as important as testing functionality. Tests
 * that are supposed to exit with an error must be annotated with the attribute
 * \code
 * // [[OutputRegex, The regex that should be found in the output]]
 * \endcode
 * Note that the regex only needs to be a sub-expression of the error message,
 * that is, there are implicit wildcards before and after the string.
 *
 * In order to test ASSERT's properly the test must also fail for release
 * builds. This is done by adding this macro at the beginning for the test.
 *
 * \example
 * \snippet Test_Time.cpp example_of_error_test
 */
#ifdef SPECTRE_DEBUG
#define ASSERTION_TEST() \
  do {                   \
    ERROR_TEST();        \
  } while (false)
#else
#include "Parallel/Abort.hpp"
#define ASSERTION_TEST()                                        \
  do {                                                          \
    ERROR_TEST();                                               \
    Parallel::abort("### No ASSERT tests in release mode ###"); \
  } while (false)
#endif

/*!
 * \ingroup TestingFramework
 * \brief Serializes and deserializes an object `t` of type `T`
 *
 * \example
 * To test serialization of copyable types use:
 *
 * snippet Test_PupStlCpp11.cpp example_serialize_copyable
 *
 * and for non-copyable types use:
 * snippet Test_PupStlCpp11.cpp example_serialize_non_copyable
 */
template <typename T>
T serialize_and_deserialize(const T& t) {
  static_assert(
      std::is_default_constructible<T>::value,
      "Cannot use serialize_and_deserialize if a class is not default "
      "constructible.");
  return deserialize<T>(serialize<T>(t).data());
}

/// Test for copy semantics assuming operator== is implement correctly
template <typename T, Requires<tt::has_equivalence<T>::value> = nullptr>
void test_copy_semantics(const T& a) {
  static_assert(std::is_copy_assignable<T>::value,
                "Class is not copy assignable.");
  static_assert(std::is_copy_constructible<T>::value,
                "Class is not copy constructible.");
  T b = a;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wparentheses"
  CHECK(b == a);
  // clang-tidy: intentionally not a reference to force invocation of copy
  // constructor
  const T c(a);  // NOLINT
  CHECK(c == a);
  // clang-tidy: self-assignment
  b = b;  // NOLINT
  CHECK(b == a);
#pragma GCC diagnostic pop
}

/// Test for move semantics assuming operator== is implement correctly
/// \requires `std::is_rvalue_reference<decltype(a)>::%value` is true
template <typename T, Requires<tt::has_equivalence<T>::value> = nullptr>
void test_move_semantics(T&& a, const T& comparison) {
  static_assert(std::is_rvalue_reference<decltype(a)>::value,
                "Must move into test_move_semantics");
  static_assert(std::is_nothrow_move_assignable<T>::value,
                "Class is not nothrow move assignable.");
  static_assert(std::is_nothrow_move_constructible<T>::value,
                "Class is not nothrow move constructible.");
  static_assert(std::is_default_constructible<T>::value,
                "Cannot use test_move_semantics if a class is not default "
                "constructible.");
  if (&a == &comparison or a != comparison) {
    // We use ERROR instead of ASSERT (which we normally should be using) to
    // guard against someone writing tests in Release mode where ASSERTs don't
    // show up.
    ERROR("'a' and 'comparison' must be distinct (but equal in value) objects");
  }
  T b;
  // clang-tidy: use std::forward instead of std::move
  b = std::move(a);  // NOLINT
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wparentheses"
  CHECK(b == comparison);
  T c(std::move(b));
  CHECK(c == comparison);
#pragma GCC diagnostic pop
}

// Test for iterators
template <typename Container>
void test_iterators(Container& c) {
  CHECK(std::distance(c.begin(), c.end()) ==
        static_cast<decltype(std::distance(c.begin(), c.end()))>(c.size()));
  CHECK(c.begin() == c.cbegin());
  CHECK(c.end() == c.cend());

  const auto& const_c = c;
  CHECK(std::distance(const_c.begin(), const_c.end()) ==
        static_cast<decltype(std::distance(const_c.begin(), const_c.end()))>(
            const_c.size()));
  CHECK(const_c.begin() == const_c.cbegin());
  CHECK(const_c.end() == const_c.cend());
}

// Test for reverse iterators
template <typename Container>
void test_reverse_iterators(Container& c) {
  CHECK(std::distance(c.rbegin(), c.rend()) ==
        static_cast<decltype(std::distance(c.rbegin(), c.rend()))>(c.size()));

  CHECK(c.rbegin() == c.crbegin());
  CHECK(c.rend() == c.crend());

  auto it = c.begin();
  auto rit = c.rbegin();
  auto end = c.end();
  auto rend = c.rend();
  auto cit = c.cbegin();
  auto cend = c.cend();
  auto crit = c.crbegin();
  auto crend = c.crend();

  for (size_t i = 0; i < c.size(); i++) {
    CHECK(*it == *(std::prev(rend, 1)));
    CHECK(*rit == *(std::prev(end, 1)));
    CHECK(*cit == *(std::prev(crend, 1)));
    CHECK(*crit == *(std::prev(cend, 1)));
    it++;
    rit++;
    rend--;
    end--;
    crit++;
    cit++;
    crend--;
    cend--;
  }

  const auto& const_c = c;
  CHECK(std::distance(const_c.begin(), const_c.end()) ==
        static_cast<decltype(std::distance(const_c.begin(), const_c.end()))>(
            const_c.size()));
  auto c_it = const_c.begin();
  auto c_rit = const_c.rbegin();
  auto c_end = const_c.end();
  auto c_rend = const_c.rend();
  for (size_t i = 0; i < c.size(); i++) {
    CHECK(*c_it == *(std::prev(c_rend, 1)));
    CHECK(*c_rit == *(std::prev(c_end, 1)));
    c_it++;
    c_rit++;
    c_rend--;
    c_end--;
  }
}

/*!
 * \ingroup TestingFramework
 * \brief Function to test comparison operators.  Pass values with
 * less < greater.
 */
template <typename T>
void check_cmp(const T& less, const T& greater) {
  CHECK(less == less);
  CHECK_FALSE(less == greater);
  CHECK(less != greater);
  CHECK_FALSE(less != less);
  CHECK(less < greater);
  CHECK_FALSE(greater < less);
  CHECK(greater > less);
  CHECK_FALSE(less > greater);
  CHECK(less <= greater);
  CHECK_FALSE(greater <= less);
  CHECK(greater >= less);
  CHECK_FALSE(less >= greater);
  CHECK(less <= less);
  CHECK_FALSE(less < less);
  CHECK(less >= less);
  CHECK_FALSE(less > less);
}

/*!
 * \ingroup TestingFramework
 * \brief Check a op b == c and also the op= version.
 */
#define CHECK_OP(a, op, b, c)   \
  do {                          \
    const auto& a_ = a;         \
    const auto& b_ = b;         \
    const auto& c_ = c;         \
    CHECK(a_ op b_ == c_);      \
    auto f = a_;                \
    CHECK((f op## = b_) == c_); \
    CHECK(f == c_);             \
  } while (false)

/*!
 * \ingroup TestingFramework
 * \brief Get the streamed output `c` as a `std::string`
 */
template <typename Container>
std::string get_output(const Container& c) {
  std::ostringstream os;
  os << c;
  return os.str();
}

/*!
 * \ingroup TestingFramework
 * \brief Alternative to Catch's CAPTURE that prints more digits.
 */
#define CAPTURE_PRECISE(variable)                                    \
  INFO(std::scientific << std::setprecision(18) << #variable << ": " \
                       << (variable));

/*!
 * \ingroup TestingFramework
 * \brief Calculates the derivative of an Invocable at a point x - represented
 * by an array of doubles - in the domain of `map` with a sixth-order finite
 * difference method.
 *
 * \details Intended for use with CoordinateMaps taking the domain {xi,eta,zeta}
 * to the range {x,y,z}. This function calculates the derivative along the
 * direction given by `direction` with a step size of `h`.
 *
 * \requires direction be between 0 and VolumeDim
 */
template <typename Invocable, size_t VolumeDim>
std::array<double, VolumeDim> numerical_derivative(
    const Invocable& map, const std::array<double, VolumeDim>& x,
    const size_t direction, const double delta) {
  ASSERT(0 <= direction and direction < VolumeDim,
         "Trying to take derivative along axis " << direction);

  const auto dx = [direction, delta]() {
    auto d = make_array<3>(0.);
    gsl::at(d, direction) = delta;
    return d;
  }();

  const std::array<double, VolumeDim> x_1ahead = x + dx;
  const std::array<double, VolumeDim> x_2ahead = x_1ahead + dx;
  const std::array<double, VolumeDim> x_3ahead = x_2ahead + dx;
  const std::array<double, VolumeDim> x_1behind = x - dx;
  const std::array<double, VolumeDim> x_2behind = x_1behind - dx;
  const std::array<double, VolumeDim> x_3behind = x_2behind - dx;
  return (1.0 / (60.0 * delta)) * map(x_3ahead) +
         (-3.0 / (20.0 * delta)) * map(x_2ahead) +
         (0.75 / delta) * map(x_1ahead) + (-0.75 / delta) * map(x_1behind) +
         (3.0 / (20.0 * delta)) * map(x_2behind) +
         (-1.0 / (60.0 * delta)) * map(x_3behind);
}
