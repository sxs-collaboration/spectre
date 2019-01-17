// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Code to wrap or improve the Catch testing framework used for unit tests.

#pragma once

#include <catch.hpp>
#include <csignal>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "ErrorHandling/Error.hpp"
#include "Parallel/Abort.hpp"
#include "Parallel/Exit.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
// The macro SPECTRE_TEST_REGISTER_FUNCTION is defined inside the
// add_test_library CMake function. It is used to make a call into a translation
// unit so that static variables for Catch are properly initialized.
#ifdef SPECTRE_TEST_REGISTER_FUNCTION
void SPECTRE_TEST_REGISTER_FUNCTION() noexcept {}  // NOLINT
#endif  // SPECTRE_TEST_REGISTER_FUNCTION
/// \endcond

namespace TestHelpers_detail {
template <typename T>
std::string format_capture_precise(const T& t) noexcept {
  std::ostringstream os;
  os << std::scientific << std::setprecision(18) << t;
  return os.str();
}
}  // namespace TestHelpers_detail

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Alternative to Catch's CAPTURE that prints more digits.
 */
#define CAPTURE_PRECISE(variable) \
  INFO(#variable << ": "          \
                 << TestHelpers_detail::format_capture_precise(variable))

/*!
 * \ingroup TestingFrameworkGroup
 * \brief A replacement for Catch's TEST_CASE that silences clang-tidy warnings
 */
#define SPECTRE_TEST_CASE(m, n) TEST_CASE(m, n)  // NOLINT

/*!
 * \ingroup TestingFrameworkGroup
 * \brief A similar to Catch's REQUIRE statement, but can be used in tests that
 * spawn several chares with possibly complex interaction between the chares.
 */
#define SPECTRE_PARALLEL_REQUIRE(expr)                                  \
  do {                                                                  \
    if (not(expr)) {                                                    \
      ERROR("\nFailed comparison: " << #expr << "\nLine: " << __LINE__  \
                                    << "\nFile: " << __FILE__ << "\n"); \
    }                                                                   \
  } while (false)

/*!
 * \ingroup TestingFrameworkGroup
 * \brief A similar to Catch's REQUIRE_FALSE statement, but can be used in tests
 * that spawn several chares with possibly complex interaction between the
 * chares.
 */
#define SPECTRE_PARALLEL_REQUIRE_FALSE(expr)                            \
  do {                                                                  \
    if ((expr)) {                                                       \
      ERROR("\nFailed comparison: " << #expr << "\nLine: " << __LINE__  \
                                    << "\nFile: " << __FILE__ << "\n"); \
    }                                                                   \
  } while (false)

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Set a default tolerance for floating-point number comparison
 *
 * \details
 * Catch's default (relative) tolerance for comparing floating-point numbers is
 * `std::numeric_limits<float>::%epsilon() * 100`, or roughly \f$10^{-5}\f$.
 * This tolerance is too loose for checking many scientific algorithms that
 * rely on double precision floating-point accuracy, so we provide a tighter
 * tighter tolerance through the `approx` static object.
 *
 * \example
 * \snippet Test_TestingFramework.cpp approx_test
 */
// clang-tidy: static object creation may throw exception
static Approx approx =                                          // NOLINT
    Approx::custom()                                            // NOLINT
        .epsilon(std::numeric_limits<double>::epsilon() * 100)  // NOLINT
        .scale(1.0);                                            // NOLINT

/*!
 * \ingroup TestingFrameworkGroup
 * \brief A wrapper around Catch's CHECK macro that checks approximate
 * equality of entries in iterable containers.  For maplike
 * containers, keys are checked for strict equality and values are
 * checked for approximate equality.
 */
#define CHECK_ITERABLE_APPROX(a, b)                                          \
  do {                                                                       \
    INFO(__FILE__ ":" + std::to_string(__LINE__) + ": " #a " == " #b);       \
    check_iterable_approx<std::common_type_t<                                \
        std::decay_t<decltype(a)>, std::decay_t<decltype(b)>>>::apply(a, b); \
  } while (false)

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Same as `CHECK_ITERABLE_APPROX` with user-defined Approx.
 *  The third argument should be of type `Approx`.
 */
#define CHECK_ITERABLE_CUSTOM_APPROX(a, b, appx)                             \
  do {                                                                       \
    INFO(__FILE__ ":" + std::to_string(__LINE__) + ": " #a " == " #b);       \
    check_iterable_approx<std::common_type_t<                                \
        std::decay_t<decltype(a)>, std::decay_t<decltype(b)>>>::apply(a, b,  \
                                                                      appx); \
  } while (false)

/// \cond HIDDEN_SYMBOLS
template <typename T, typename = std::nullptr_t>
struct check_iterable_approx {
  // clang-tidy: non-const reference
  static void apply(const T& a, const T& b, Approx& appx = approx) {  // NOLINT
    CAPTURE_PRECISE(a);
    CAPTURE_PRECISE(b);
    CHECK(a == appx(b));
  }
};

template <typename T>
struct check_iterable_approx<std::complex<T>, std::nullptr_t> {
  // clang-tidy: non-const reference
  static void apply(const std::complex<T>& a, const std::complex<T>& b,
                    Approx& appx = approx) {  // NOLINT
    check_iterable_approx<T>::apply(real(a), real(b), appx);
    check_iterable_approx<T>::apply(imag(a), imag(b), appx);
  }
};

template <typename T>
struct check_iterable_approx<
    T, Requires<not tt::is_maplike_v<T> and tt::is_iterable_v<T> and
                not tt::is_a_v<std::unordered_set, T>>> {
  // clang-tidy: non-const reference
  static void apply(const T& a, const T& b, Approx& appx = approx) {  // NOLINT
    CAPTURE_PRECISE(a);
    CAPTURE_PRECISE(b);
    auto a_it = a.begin();
    auto b_it = b.begin();
    CHECK(a_it != a.end());
    CHECK(b_it != b.end());
    while (a_it != a.end() and b_it != b.end()) {
      check_iterable_approx<std::decay_t<decltype(*a_it)>>::apply(*a_it, *b_it,
                                                                  appx);
      ++a_it;
      ++b_it;
    }
    {
      INFO("Iterable is longer in first argument than in second argument");
      CHECK(a_it == a.end());
    }
    {
      INFO("Iterable is shorter in first argument than in second argument");
      CHECK(b_it == b.end());
    }
  }
};

template <typename T>
struct check_iterable_approx<T, Requires<tt::is_a_v<std::unordered_set, T>>> {
  // clang-tidy: non-const reference
  static void apply(const T& a, const T& b,
                    Approx& /*appx*/ = approx) {  // NOLINT
    CAPTURE_PRECISE(a);
    CAPTURE_PRECISE(b);
    // Approximate comparison of unordered sets is difficult
    CHECK(a == b);
  }
};

template <typename T>
struct check_iterable_approx<
    T, Requires<tt::is_maplike_v<T> and tt::is_iterable_v<T>>> {
  // clang-tidy: non-const reference
  static void apply(const T& a, const T& b, Approx& appx = approx) {  // NOLINT
    CAPTURE_PRECISE(a);
    CAPTURE_PRECISE(b);
    for (const auto& kv : a) {
      const auto& key = kv.first;
      try {
        const auto& a_value = kv.second;
        const auto& b_value = b.at(key);
        CAPTURE(key);
        check_iterable_approx<std::decay_t<decltype(a_value)>>::apply(
            a_value, b_value, appx);
      } catch (const std::out_of_range&) {
        INFO("Missing key in second container: " << key);
        CHECK(false);
      }
    }

    for (const auto& kv : b) {
      const auto& key = kv.first;
      try {
        a.at(key);
        // We've checked that the values match above.
      } catch (const std::out_of_range&) {
        INFO("Missing key in first container: " << key);
        CHECK(false);
      }
    }
  }
};

#define CHECK_MATRIX_APPROX(a, b)                                            \
  do {                                                                       \
    INFO(__FILE__ ":" + std::to_string(__LINE__) + ": " #a " == " #b);       \
    check_matrix_approx<std::common_type_t<                                  \
        std::decay_t<decltype(a)>, std::decay_t<decltype(b)>>>::apply(a, b); \
  } while (false)

#define CHECK_MATRIX_CUSTOM_APPROX(a, b, appx)                               \
  do {                                                                       \
    INFO(__FILE__ ":" + std::to_string(__LINE__) + ": " #a " == " #b);       \
    check_matrix_approx<std::common_type_t<                                  \
        std::decay_t<decltype(a)>, std::decay_t<decltype(b)>>>::apply(a, b,  \
                                                                      appx); \
  } while (false)

template <typename M>
struct check_matrix_approx {
  // clang-tidy: non-const reference
  static void apply(const M& a, const M& b,
                    Approx& appx = approx) {  // NOLINT
    // This implementation is for a column-major matrix. It does not trivially
    // generalize to a row-major matrix because the iterator
    // `blaze::DynamicMatrix<T, SO>::cbegin(i)` traverses either a row or a
    // column, and takes its index as argument.
    CHECK(a.columns() == b.columns());
    for (size_t j = 0; j < a.columns(); j++) {
      CAPTURE_PRECISE(a);
      CAPTURE_PRECISE(b);
      auto a_it = a.cbegin(j);
      auto b_it = b.cbegin(j);
      CHECK(a_it != a.cend(j));
      CHECK(b_it != b.cend(j));
      while (a_it != a.cend(j) and b_it != b.cend(j)) {
        check_iterable_approx<std::decay_t<decltype(*a_it)>>::apply(
            *a_it, *b_it, appx);
        ++a_it;
        ++b_it;
      }
      {
        INFO("Column " << j
                       << " of the first matrix is longer than that of the "
                          "second matrix.");
        CHECK(a_it == a.end(j));
      }
      {
        INFO("Column " << j
                       << " of the first matrix is shorter than that of the "
                          "second matrix.");
        CHECK(b_it == b.end(j));
      }
    }
  }
};
/// \endcond

/// \cond HIDDEN_SYMBOLS
[[noreturn]] inline void spectre_testing_signal_handler(int /*signal*/) {
  Parallel::exit();
}
/// \endcond

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Mark a test as checking a call to ERROR
 *
 * \details
 * In order to properly handle aborting with Catch versions newer than 1.6.1
 * we must install a signal handler after Catch does, which means inside the
 * SPECTRE_TEST_CASE itself. The ERROR_TEST() macro should be the first line in
 * the SPECTRE_TEST_CASE.
 *
 * \example
 * \snippet Test_TestingFramework.cpp error_test
 */
#define ERROR_TEST()                                      \
  do {                                                    \
    std::signal(SIGABRT, spectre_testing_signal_handler); \
  } while (false)

/*!
 * \ingroup TestingFrameworkGroup
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
