// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Commonly used routines, functions and definitions shared amongst unit tests

#pragma once

#include <catch.hpp>
#include <charm++.h>
#include <csignal>
#include <pup.h>

#include "ErrorHandling/Error.hpp"
#include "Parallel/Abort.hpp"
#include "Parallel/Serialize.hpp"
#include "Utilities/TypeTraits.hpp"

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
 * TEST_CASE itself. The ERROR_TEST() macro should be the first line in the
 * TEST_CASE.
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
 * snippet Test_Time.cpp example_of_error_test
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
  return deserialize<T>(serialize<T>(t).data());
}

/// Test for copy semantics assuming operator== is implement correctly
template <typename T,
          typename std::enable_if<tt::has_equivalence<T>::value, int>::type = 0>
void test_copy_semantics(const T& a) {
  static_assert(std::is_copy_assignable<T>::value,
                "Class is not copy assignable.");
  static_assert(std::is_copy_constructible<T>::value,
                "Class is not copy constructible.");
  T b = a;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wparentheses"
  CHECK(b == a);
  // Intentionally not a reference to force invocation of copy constructor
  const T c(a);  // NOLINT
  CHECK(c == a);
  // Check self-assignment
  b = b;  // NOLINT
  CHECK(b == a);
#pragma GCC diagnostic pop
}

/// Test for move semantics assuming operator== is implement correctly
template <typename T,
          typename std::enable_if<tt::has_equivalence<T>::value, int>::type = 0>
void test_move_semantics(T&& a, const T& comparison) {
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
  b = std::move(a);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wparentheses"
  CHECK(b == comparison);
  T c(std::move(b));
  CHECK(c == comparison);
#pragma GCC diagnostic pop
}

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
